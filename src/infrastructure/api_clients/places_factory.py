# src/infrastructure/api_clients/places_factory.py

from typing import Optional
from .google_places.client import GooglePlacesClient
from ...config.settings import get_settings


class PlacesClientFactory:
    """Factory to create Google Places API client with fallback to mock"""

    @staticmethod
    def create_client(provider: str = "auto", cache_adapter=None, **kwargs):
        """Create Google Places API client based on provider and available keys"""

        settings = get_settings()

        # Debug logging
        print(f"DEBUG FACTORY: provider={provider}")
        print(f"DEBUG FACTORY: google_places_api_key={'[SET]' if settings.api.google_places_api_key else '[NOT SET]'}")

        if provider == "auto":
            # Auto-select: Google Places with API key, or Mock if no key
            if settings.api.google_places_api_key:
                print("DEBUG FACTORY: Auto-selected Google Places with real API")
                return GooglePlacesClient(
                    api_key=settings.api.google_places_api_key,
                    cache_adapter=cache_adapter,
                    use_mock=False,
                    **kwargs
                )
            else:
                print("DEBUG FACTORY: Auto-selected Google Places with mock data (no API key)")
                return GooglePlacesClient(
                    cache_adapter=cache_adapter,
                    use_mock=True,
                    **kwargs
                )

        elif provider == "google":
            print("DEBUG FACTORY: Creating GooglePlacesClient")
            return GooglePlacesClient(
                api_key=settings.api.google_places_api_key,
                cache_adapter=cache_adapter,
                use_mock=not settings.api.google_places_api_key,
                **kwargs
            )

        elif provider == "mock":
            print("DEBUG FACTORY: Creating Mock GooglePlacesClient")
            return GooglePlacesClient(
                cache_adapter=cache_adapter,
                use_mock=True,
                **kwargs
            )

        else:
            raise ValueError(
                f"Unknown places provider: {provider}. Valid options: 'auto', 'google', 'mock'"
            )