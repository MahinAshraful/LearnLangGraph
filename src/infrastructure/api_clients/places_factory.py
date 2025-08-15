# src/infrastructure/api_clients/places_factory.py

from typing import Optional
from .google_places.client import GooglePlacesClient
from .foursquare.client import FoursquareClient
from ...config.settings import get_settings


class PlacesClientFactory:
    """Factory to create the appropriate places API client"""

    @staticmethod
    def create_client(provider: str = "auto", cache_adapter=None, **kwargs):
        """Create places API client based on provider and available keys"""

        settings = get_settings()

        # Debug logging
        print(f"DEBUG FACTORY: provider={provider}")
        print(f"DEBUG FACTORY: google_places_api_key={'[SET]' if settings.api.google_places_api_key else '[NOT SET]'}")
        print(f"DEBUG FACTORY: foursquare_api_key={'[SET]' if settings.api.foursquare_api_key else '[NOT SET]'}")

        if provider == "auto":
            # Priority: Google Places > Foursquare > Mock
            if settings.api.google_places_api_key:
                print("DEBUG FACTORY: Auto-selected google")
                return GooglePlacesClient(
                    api_key=settings.api.google_places_api_key,
                    cache_adapter=cache_adapter,
                    use_mock=False,
                    **kwargs
                )
            elif settings.api.foursquare_api_key:
                print("DEBUG FACTORY: Auto-selected foursquare")
                return FoursquareClient(
                    api_key=settings.api.foursquare_api_key,
                    cache_adapter=cache_adapter,
                    **kwargs
                )
            else:
                print("DEBUG FACTORY: Auto-selected mock (no API keys)")
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

        elif provider == "foursquare":
            print(f"DEBUG FACTORY: Creating FoursquareClient")
            return FoursquareClient(
                api_key=settings.api.foursquare_api_key,
                cache_adapter=cache_adapter,
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
                f"Unknown places provider: {provider}. Valid options: 'auto', 'google', 'foursquare', 'mock'")