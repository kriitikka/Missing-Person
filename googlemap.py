import googlemaps

gmaps = googlemaps.Client(key="your_google_maps_api_key")
geocode_result = gmaps.reverse_geocode((latitude, longitude))
location = geocode_result[0]['formatted_address'] if geocode_result else "Unknown"
