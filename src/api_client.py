# src/api_client.py
from settings import Config
import requests
import datetime

def api_call_all_sites():
    """
    Makes authenticated GET requests to various site APIs.
    Uses Config class for credentials and headers.
    """
    base_url = 'https://wrightbus-XX.odosolutions.com/api/v1/downloads'
    headers = Config.get_auth_headers()  # Replaces get_auth_headers(USERNAME, PASSWORD)
    
    today = datetime.date.today()
    last_sunday = today - datetime.timedelta(days=(today.weekday() + 1) % 7 + 7)
    last_monday = last_sunday + datetime.timedelta(days=1)

    valid_entries_all_sites = []

    for site_num in range(1, 13):
        site_str = f'{site_num:02d}'
        url = base_url.replace('XX', site_str)

        try:
            response = requests.get(url, headers=headers, timeout=30)
            response.raise_for_status()
            
            for entry in response.json():
                if isinstance(entry, dict) and 'minDateTime' in entry:
                    try:
                        min_date = datetime.datetime.strptime(
                            entry['minDateTime'], 
                            '%Y-%m-%dT%H:%M:%SZ'
                        ).date()
                        
                        if min_date in (last_sunday, last_monday):
                            valid_entries_all_sites.append({
                                'site': site_str,
                                'scheduleId': entry.get('scheduledById'),
                                'name': entry.get('name'),
                                'id': entry.get('id')
                            })
                    except ValueError as e:
                        print(f"Date parse error (site {site_str}): {e}")

        except requests.exceptions.RequestException as e:
            print(f"API error (site {site_str}): {e}")

    return valid_entries_all_sites