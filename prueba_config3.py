from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Protocol

from openai import APIError, OpenAI

from dataclasses import dataclass, field
from pathlib import Path

import yaml


@dataclass(frozen=True, slots=True)
class ModelMessage:
    role: str
    content: str


class ModelAdapter(Protocol):
    def complete(self, messages: list[ModelMessage]) -> str:
        raise NotImplementedError

class OpenAIModelAdapter:
    def __init__(
        self,
        *,
        model: str,
        api_base: str,
        api_key: str,
        temperature: float,
    ) -> None:
        self.model = model
        self.api_base = api_base.rstrip("/")
        self.api_key = api_key
        self.temperature = temperature

    def complete(self, messages: list[ModelMessage]) -> str:
        if not self.api_key:
            raise RuntimeError("Missing model API key in config.agent.api_key.")

        client = OpenAI(
            api_key=self.api_key,
            base_url=self.api_base,
        )

        try:
            response = client.chat.completions.create(
                model=self.model,
                messages=[{"role": message.role, "content": message.content} for message in messages],
                temperature=self.temperature
            )
        except APIError as exc:
            raise RuntimeError(f"Model request failed: {exc}") from exc

        choices = response.choices or []
        if not choices:
            raise RuntimeError("Model response missing choices.")
        content = choices[0].message.content
        if not isinstance(content, str):
            raise RuntimeError("Model response missing text content.")
        return content
    

def build_model_adapter(config: AppConfig):
    return OpenAIModelAdapter(
        model=config.agent.model,
        api_base=config.agent.api_base,
        api_key=config.agent.api_key,
        temperature=config.agent.temperature,
    )


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _default_dataset_root() -> Path:
    return PROJECT_ROOT / "data" / "public" / "input"


def _default_run_output_dir() -> Path:
    return PROJECT_ROOT / "artifacts" / "runs"


@dataclass(frozen=True, slots=True)
class DatasetConfig:
    root_path: Path = field(default_factory=_default_dataset_root)


@dataclass(frozen=True, slots=True)
class AgentConfig:
    model: str = "gpt-4.1-mini"
    api_base: str = "https://api.openai.com/v1"
    api_key: str = ""
    max_steps: int = 16
    temperature: float = 0.0


@dataclass(frozen=True, slots=True)
class RunConfig:
    output_dir: Path = field(default_factory=_default_run_output_dir)
    run_id: str | None = None
    max_workers: int = 4
    task_timeout_seconds: int = 600


@dataclass(frozen=True, slots=True)
class AppConfig:
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    agent: AgentConfig = field(default_factory=AgentConfig)
    run: RunConfig = field(default_factory=RunConfig)

def _path_value(raw_value: str | None, default_value: Path) -> Path:
    if not raw_value:
        return default_value
    candidate = Path(raw_value)
    if candidate.is_absolute():
        return candidate
    return (PROJECT_ROOT / candidate).resolve()


def load_app_config(config_path: Path) -> AppConfig:
    payload = yaml.safe_load(config_path.read_text()) or {}
    dataset_defaults = DatasetConfig()
    agent_defaults = AgentConfig()
    run_defaults = RunConfig()

    dataset_payload = payload.get("dataset", {})
    agent_payload = payload.get("agent", {})
    run_payload = payload.get("run", {})

    dataset_config = DatasetConfig(
        root_path=_path_value(dataset_payload.get("root_path"), dataset_defaults.root_path),
    )
    agent_config = AgentConfig(
        model=str(agent_payload.get("model", agent_defaults.model)),
        api_base=str(agent_payload.get("api_base", agent_defaults.api_base)),
        api_key=str(agent_payload.get("api_key", agent_defaults.api_key)),
        max_steps=int(agent_payload.get("max_steps", agent_defaults.max_steps)),
        temperature=float(agent_payload.get("temperature", agent_defaults.temperature)),
    )
    raw_run_id = run_payload.get("run_id")
    run_id = run_defaults.run_id
    if raw_run_id is not None:
        normalized_run_id = str(raw_run_id).strip()
        run_id = normalized_run_id or None

    run_config = RunConfig(
        output_dir=_path_value(run_payload.get("output_dir"), run_defaults.output_dir),
        run_id=run_id,
        max_workers=int(run_payload.get("max_workers", run_defaults.max_workers)),
        task_timeout_seconds=int(run_payload.get("task_timeout_seconds", run_defaults.task_timeout_seconds)),
    )
    return AppConfig(dataset=dataset_config, agent=agent_config, run=run_config)


if __name__ == "__main__":
    ruta_config = Path("configs/nueva_config.yaml") 

    try:
        print(f"Intentando cargar configuración desde: {ruta_config.resolve()}")
        configuracion = load_app_config(ruta_config)
        
        print("\n ¡Configuración leída con éxito!")
        print(f"  -> Modelo: {configuracion.agent.model}")
        print(f"  -> API Base: {configuracion.agent.api_base}")
        print(f"  -> Temperatura: {configuracion.agent.temperature}")
        print(f"  -> Directorio de salida: {configuracion.run.output_dir}")

        print("\nConstruyendo el adaptador del modelo...")
        adaptador = build_model_adapter(configuracion)
        print("  Adaptador construido correctamente.")

        if configuracion.agent.api_key:
            print("\nEnviando mensaje de prueba a la API...")
            mensajes_prueba = [ModelMessage(role="user", content="Responde solo con '¡Conexión exitosa!'")]
            respuesta = adaptador.complete(mensajes_prueba)
            print(f" Respuesta del modelo: {respuesta}")
        else:
            print("\n No se encontró 'api_key' en la configuración. Se omite la prueba de conexión a la API.")

    except FileNotFoundError:
        print(f"Error: No se pudo encontrar el archivo de configuración en {ruta_config}")
    except Exception as e:
        print(f"Ocurrió un error inesperado: {e}")