import os
from typing import Dict, List, Optional, Any
from dataclasses import dataclass, field
from enum import Enum


class Environment(str, Enum):
    """Application environments"""
    DEVELOPMENT = "development"
    TESTING = "testing"
    STAGING = "staging"
    PRODUCTION = "production"


@dataclass
class APIConfig:
    """External API configuration"""

    # OpenAI Configuration
    openai_api_key: str = field(default_factory=lambda: os.getenv("OPENAI_API_KEY", ""))
    openai_org_id: Optional[str] = field(default_factory=lambda: os.getenv("OPENAI_ORG_ID"))

    # Google Places API (Primary Places Provider)
    google_places_api_key: str = field(default_factory=lambda: os.getenv("GOOGLE_PLACES_API_KEY", ""))

    # Weather API (OpenWeatherMap)
    weather_api_key: str = field(default_factory=lambda: os.getenv("WEATHER_API_KEY", ""))

    # Eventbrite API
    eventbrite_api_key: str = field(default_factory=lambda: os.getenv("EVENTBRITE_API_KEY", ""))

    # Rate limiting (removed Foursquare)
    api_rate_limits: Dict[str, int] = field(default_factory=lambda: {
        "google_places": 100,  # requests per minute
        "openai": 60,
        "weather": 60,
        "eventbrite": 50
    })


@dataclass
class DatabaseConfig:
    """Database configuration"""

    # Vector Database (Chroma/Pinecone)
    vector_db_type: str = field(default="chroma")  # "chroma" or "pinecone"
    chroma_host: str = field(default="localhost")
    chroma_port: int = field(default=8000)
    pinecone_api_key: str = field(default_factory=lambda: os.getenv("PINECONE_API_KEY", ""))
    pinecone_environment: str = field(default="us-west1-gcp")

    # Cache Database (Redis)
    redis_url: str = field(default_factory=lambda: os.getenv("REDIS_URL", "redis://localhost:6379"))
    redis_password: Optional[str] = field(default_factory=lambda: os.getenv("REDIS_PASSWORD"))

    # SQL Database (PostgreSQL)
    postgres_url: str = field(
        default_factory=lambda: os.getenv("DATABASE_URL", "postgresql://localhost/restaurant_rec"))

    # Connection pools
    max_connections: int = field(default=20)
    connection_timeout: int = field(default=30)


@dataclass
class LLMConfig:
    """LLM model configuration"""

    # Model selection for different tasks
    models: Dict[str, str] = field(default_factory=lambda: {
        "query_parser": "gpt-3.5-turbo",  # Fast, cheap for parsing
        "reasoning": "gpt-4",  # Smart reasoning when needed
        "embeddings": "text-embedding-3-large"  # For user/restaurant embeddings
    })

    # Model parameters
    temperature: Dict[str, float] = field(default_factory=lambda: {
        "query_parser": 0.1,  # Low temperature for consistent parsing
        "reasoning": 0.3  # Slightly higher for creative reasoning
    })

    # Token limits
    max_tokens: Dict[str, int] = field(default_factory=lambda: {
        "query_parser": 500,
        "reasoning": 1000
    })

    # Cost optimization
    enable_caching: bool = field(default=True)
    cache_ttl_seconds: int = field(default=3600)  # 1 hour


@dataclass
class RecommendationConfig:
    """Recommendation algorithm configuration"""

    # Scoring weights (50/30/15/5 algorithm)
    scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        "preference_match": 0.50,  # 50% user preference matching
        "context_relevance": 0.30,  # 30% context (weather, time, etc.)
        "quality_score": 0.15,  # 15% restaurant quality
        "boost_score": 0.05  # 5% special boosts
    })

    # Search parameters
    default_search_radius_km: float = field(default=10.0)
    max_candidates: int = field(default=50)  # Maximum restaurants to score
    default_results: int = field(default=10)  # Default number of results
    min_restaurant_rating: float = field(default=3.5)
    min_review_count: int = field(default=10)

    # Personalization
    enable_collaborative_filtering: bool = field(default=True)
    similar_users_count: int = field(default=5)
    min_user_activities: int = field(default=3)  # Minimum activities for personalization

    # Caching
    cache_recommendations: bool = field(default=True)
    recommendation_cache_ttl: int = field(default=1800)  # 30 minutes

    # Smart reasoning triggers
    complexity_threshold: float = field(default=0.5)  # When to use GPT-4 reasoning
    enable_smart_reasoning: bool = field(default=True)


@dataclass
class CacheConfig:
    """Caching configuration"""

    # Cache TTLs by type (seconds)
    ttl: Dict[str, int] = field(default_factory=lambda: {
        "user_preferences": 3600,  # 1 hour
        "restaurant_data": 1800,  # 30 minutes
        "google_places": 900,  # 15 minutes (API data changes)
        "weather_data": 600,  # 10 minutes
        "query_results": 1800,  # 30 minutes
        "embeddings": 86400  # 24 hours
    })

    # Cache keys configuration
    key_prefix: str = field(default="restaurant_rec")
    version: str = field(default="v1")

    # Cache behavior
    enable_write_through: bool = field(default=True)
    enable_read_through: bool = field(default=True)
    max_cache_size_mb: int = field(default=512)


@dataclass
class MonitoringConfig:
    """Monitoring and logging configuration"""

    # Logging
    log_level: str = field(default="INFO")
    log_format: str = field(default="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    log_to_file: bool = field(default=True)
    log_file_path: str = field(default="logs/restaurant_rec.log")

    # Metrics
    enable_metrics: bool = field(default=True)
    metrics_port: int = field(default=9090)

    # Health checks
    health_check_timeout: int = field(default=30)

    # Performance tracking
    track_api_calls: bool = field(default=True)
    track_cache_performance: bool = field(default=True)
    track_recommendation_quality: bool = field(default=True)


@dataclass
class FeatureFlags:
    """Feature flags for gradual rollouts"""

    # Core features
    enable_personalization: bool = field(default=True)
    enable_collaborative_filtering: bool = field(default=True)
    enable_smart_reasoning: bool = field(default=True)

    # Context features
    enable_weather_context: bool = field(default=False)  # Not implemented yet
    enable_event_context: bool = field(default=False)  # Not implemented yet

    # Advanced features
    enable_social_features: bool = field(default=False)  # Future feature
    enable_real_time_popularity: bool = field(default=True)
    enable_dietary_filtering: bool = field(default=True)

    # System features
    enable_a_b_testing: bool = field(default=True)
    enable_analytics: bool = field(default=True)
    enable_caching: bool = field(default=True)


@dataclass
class Settings:
    """Main application settings"""

    # Environment
    environment: Environment = field(default=Environment.DEVELOPMENT)
    debug: bool = field(default=False)
    testing: bool = field(default=False)

    # API Configuration
    host: str = field(default="localhost")
    port: int = field(default=8000)
    api_prefix: str = field(default="/api/v1")

    # Component configs
    api: APIConfig = field(default_factory=APIConfig)
    database: DatabaseConfig = field(default_factory=DatabaseConfig)
    llm: LLMConfig = field(default_factory=LLMConfig)
    recommendation: RecommendationConfig = field(default_factory=RecommendationConfig)
    cache: CacheConfig = field(default_factory=CacheConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    features: FeatureFlags = field(default_factory=FeatureFlags)

    # Security
    secret_key: str = field(default_factory=lambda: os.getenv("SECRET_KEY", "dev-secret-key"))
    allowed_origins: List[str] = field(default_factory=lambda: ["http://localhost:3000", "http://localhost:8000"])

    @classmethod
    def from_env(cls) -> "Settings":
        """Create settings from environment variables"""
        environment = Environment(os.getenv("ENVIRONMENT", "development"))

        settings = cls(
            environment=environment,
            debug=os.getenv("DEBUG", "false").lower() == "true",
            testing=os.getenv("TESTING", "false").lower() == "true",
            host=os.getenv("HOST", "localhost"),
            port=int(os.getenv("PORT", "8000"))
        )

        # Override based on environment
        if environment == Environment.PRODUCTION:
            settings.debug = False
            settings.monitoring.log_level = "WARNING"
            settings.cache.ttl["query_results"] = 3600  # Longer cache in prod

        elif environment == Environment.TESTING:
            settings.cache.ttl = {k: 60 for k in settings.cache.ttl.keys()}  # Short cache in tests
            settings.recommendation.cache_recommendations = False

        return settings


# Global settings instance
_settings = None


def get_settings() -> Settings:
    """Get global settings instance"""
    global _settings
    if _settings is None:
        _settings = Settings.from_env()
    return _settings


def reload_settings():
    """Force reload settings from environment"""
    global _settings
    _settings = Settings.from_env()
    return _settings