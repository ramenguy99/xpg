"""YAML configuration loading for tracer setup."""

from pathlib import Path
from typing import Any, Callable, Dict, Union

import yaml

from ._global import init_tracer
from ._queue import QueueFullPolicy
from ._sqlite_subscriber import SqliteConfig, SqliteSubscriber
from ._subscriber import Subscriber
from ._tcp_subscriber import TcpSubscriber
from ._tracer import Tracer


class ConfigError(Exception):
    """Raised when configuration is invalid."""


# Registry of subscriber type factories
_subscriber_factories: Dict[str, Callable[[Dict[str, Any]], Subscriber]] = {}


def register_subscriber_type(
    type_name: str,
    factory: Callable[[Dict[str, Any]], Subscriber],
) -> None:
    """Register a custom subscriber type factory.

    Args:
        type_name: The type name used in YAML config.
        factory: Factory function that takes config dict and returns Subscriber.

    """
    _subscriber_factories[type_name] = factory


def _parse_queue_full_policy(value: str) -> QueueFullPolicy:
    """Parse queue full policy from string.

    Args:
        value: Policy string ("drop" or "wait").

    Returns:
        QueueFullPolicy enum value.

    Raises:
        ConfigError: If the value is invalid.

    """
    value_lower = value.lower()
    if value_lower == "drop":
        return QueueFullPolicy.DROP
    elif value_lower == "wait":
        return QueueFullPolicy.WAIT
    else:
        raise ConfigError(f"Invalid queue_full_policy: {value}. Must be 'drop' or 'wait'.")


def _create_sqlite_subscriber(config: Dict[str, Any]) -> SqliteSubscriber:
    """Create a SqliteSubscriber from config.

    Args:
        config: Subscriber configuration dict.

    Returns:
        Configured SqliteSubscriber.

    Raises:
        ConfigError: If required fields are missing.

    """
    if "db_path" not in config:
        raise ConfigError("SqliteSubscriber requires 'db_path'")

    db_path = config["db_path"]
    patterns = config.get("patterns")
    queue_size = config.get("queue_size", 10000)
    queue_full_policy = _parse_queue_full_policy(config.get("queue_full_policy", "drop"))
    verbose = config.get("verbose", False)

    # Build SqliteConfig from nested sqlite_config dict or top-level keys
    sqlite_config_dict = config.get("sqlite_config")
    if sqlite_config_dict is not None:
        sqlite_config = SqliteConfig(
            journal_mode=sqlite_config_dict.get("journal_mode", "wal"),
            synchronous=sqlite_config_dict.get("synchronous", "normal"),
            wal_autocheckpoint=sqlite_config_dict.get("wal_autocheckpoint", 16384),
            page_size=sqlite_config_dict.get("page_size"),
            cache_size=sqlite_config_dict.get("cache_size"),
        )
    else:
        sqlite_config = None

    return SqliteSubscriber(
        db_path=db_path,
        patterns=patterns,
        queue_size=queue_size,
        queue_full_policy=queue_full_policy,
        sqlite_config=sqlite_config,
        verbose=verbose,
    )


def _create_tcp_subscriber(config: Dict[str, Any]) -> TcpSubscriber:
    """Create a TcpSubscriber from config.

    Args:
        config: Subscriber configuration dict.

    Returns:
        Configured TcpSubscriber.

    Raises:
        ConfigError: If required fields are missing.

    """
    if "host" not in config:
        raise ConfigError("TcpSubscriber requires 'host'")
    if "port" not in config:
        raise ConfigError("TcpSubscriber requires 'port'")

    host = config["host"]
    port = config["port"]
    patterns = config.get("patterns")
    connect_timeout = config.get("connect_timeout", 5.0)
    reconnect_delay = config.get("reconnect_delay", 1.0)
    queue_size = config.get("queue_size", 10000)
    queue_full_policy = _parse_queue_full_policy(config.get("queue_full_policy", "drop"))
    verbose = config.get("verbose", False)

    return TcpSubscriber(
        host=host,
        port=port,
        patterns=patterns,
        connect_timeout=connect_timeout,
        reconnect_delay=reconnect_delay,
        queue_size=queue_size,
        queue_full_policy=queue_full_policy,
        verbose=verbose,
    )


# Register built-in subscriber types
register_subscriber_type("sqlite", _create_sqlite_subscriber)
register_subscriber_type("tcp", _create_tcp_subscriber)


def _create_subscriber(config: Dict[str, Any]) -> Subscriber:
    """Create a subscriber from configuration.

    Args:
        config: Subscriber configuration dict.

    Returns:
        Configured Subscriber instance.

    Raises:
        ConfigError: If configuration is invalid.

    """
    if "type" not in config:
        raise ConfigError("Subscriber configuration requires 'type'")

    sub_type = config["type"]
    factory = _subscriber_factories.get(sub_type)

    if factory is None:
        raise ConfigError(
            f"Unknown subscriber type: {sub_type}. Available types: {list(_subscriber_factories.keys())}"
        )

    return factory(config)


def load_config(config_path: Union[str, Path]) -> Tracer:
    """Load tracer configuration from a YAML file.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        Configured Tracer instance with subscribers registered.

    Raises:
        ConfigError: If configuration is invalid.
        FileNotFoundError: If config file doesn't exist.

    """
    path = Path(config_path)
    with path.open("r") as f:
        yaml_content = f.read()

    return load_config_from_string(yaml_content)


def load_config_from_string(yaml_string: str) -> Tracer:
    """Load tracer configuration from a YAML string.

    Args:
        yaml_string: YAML configuration string.

    Returns:
        Configured Tracer instance with subscribers registered.

    Raises:
        ConfigError: If configuration is invalid.

    """
    try:
        config = yaml.safe_load(yaml_string)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML: {e}") from e

    if config is None:
        config = {}

    if "tracer" not in config:
        raise ConfigError("Configuration must have 'tracer' key")

    tracer_config = config["tracer"]
    if tracer_config is None:
        tracer_config = {}

    tracer = Tracer()

    subscribers_config = tracer_config.get("subscribers", [])
    if subscribers_config is None:
        subscribers_config = []

    for sub_config in subscribers_config:
        subscriber = _create_subscriber(sub_config)
        tracer.register_subscriber(subscriber)

    return tracer


def init_from_config(config_path: Union[str, Path]) -> Tracer:
    """Initialize global tracer from a YAML configuration file.

    This loads the configuration and sets up the global tracer instance.

    Args:
        config_path: Path to the YAML configuration file.

    Returns:
        The configured global Tracer instance.

    Raises:
        ConfigError: If configuration is invalid.
        FileNotFoundError: If config file doesn't exist.

    """
    # First initialize the global tracer
    global_tracer = init_tracer()

    # Load config to get subscribers
    path = Path(config_path)
    with path.open("r") as f:
        yaml_content = f.read()

    try:
        config = yaml.safe_load(yaml_content)
    except yaml.YAMLError as e:
        raise ConfigError(f"Invalid YAML: {e}") from e

    if config is None:
        config = {}

    if "tracer" not in config:
        raise ConfigError("Configuration must have 'tracer' key")

    tracer_config = config["tracer"]
    if tracer_config is None:
        tracer_config = {}

    subscribers_config = tracer_config.get("subscribers", [])
    if subscribers_config is None:
        subscribers_config = []

    for sub_config in subscribers_config:
        subscriber = _create_subscriber(sub_config)
        global_tracer.register_subscriber(subscriber)

    return global_tracer
