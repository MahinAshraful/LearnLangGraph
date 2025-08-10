from typing import Optional
from .google_places.client import GooglePlacesClient
from .foursquare.client import FoursquareClient
from ...config.settings import get_settings


class PlacesClientFactory:
    """Factory to create the appropriate places API client"""

    @staticmethod
    def create_client(provider: str = "auto", cache_adapter=None, **kwargs):
        settings = get_settings()

        # DEBUG PRINTS
        print(f"DEBUG FACTORY: provider={provider}")
        print(f"DEBUG FACTORY: settings.api.foursquare_api_key={repr(settings.api.foursquare_api_key)}")

        if provider == "auto":
            if settings.api.foursquare_api_key:
                provider = "foursquare"
                print(f"DEBUG FACTORY: Auto-selected foursquare")
            elif settings.api.google_places_api_key:
                provider = "google"
                print(f"DEBUG FACTORY: Auto-selected google")
            else:
                provider = "mock"
                print(f"DEBUG FACTORY: Auto-selected mock")

        if provider == "foursquare":
            print(f"DEBUG FACTORY: Creating FoursquareClient with key={repr(settings.api.foursquare_api_key)}")
            return FoursquareClient(
                api_key=settings.api.foursquare_api_key,
                cache_adapter=cache_adapter,
                **kwargs
            )
        # ... rest of method
        elif provider == "google":
            return GooglePlacesClient(
                api_key=settings.api.google_places_api_key,
                cache_adapter=cache_adapter,
                use_mock=False,
                **kwargs
            )
        elif provider == "mock":
            return GooglePlacesClient(
                cache_adapter=cache_adapter,
                use_mock=True,
                **kwargs
            )
        else:
            raise ValueError(f"Unknown places provider: {provider}")