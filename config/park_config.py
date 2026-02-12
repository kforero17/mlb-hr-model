from dataclasses import dataclass


@dataclass(frozen=True)
class ParkInfo:
    name: str
    lat: float
    lon: float
    elevation_ft: int
    roof: str
    cf_bearing_deg: float


PARKS: dict[str, ParkInfo] = {
    "NYY": ParkInfo("Yankee Stadium", 40.8296, -73.9262, 20, "open", 87.0),
    "BOS": ParkInfo("Fenway Park", 42.3467, -71.0972, 20, "open", 68.0),
    "TOR": ParkInfo("Rogers Centre", 43.6414, -79.3894, 250, "retractable", 0.0),
    "BAL": ParkInfo("Oriole Park at Camden Yards", 39.2839, -76.6217, 30, "open", 68.0),
    "TB":  ParkInfo("Tropicana Field", 27.7683, -82.6534, 42, "dome", 0.0),
    "CLE": ParkInfo("Progressive Field", 41.4962, -81.6852, 653, "open", 36.0),
    "CWS": ParkInfo("Guaranteed Rate Field", 41.8300, -87.6339, 595, "open", 22.0),
    "DET": ParkInfo("Comerica Park", 42.3390, -83.0485, 600, "open", 39.0),
    "KC":  ParkInfo("Kauffman Stadium", 39.0517, -94.4803, 750, "open", 90.0),
    "MIN": ParkInfo("Target Field", 44.9817, -93.2775, 841, "open", 22.0),
    "HOU": ParkInfo("Minute Maid Park", 29.7573, -95.3554, 42, "retractable", 72.0),
    "LAA": ParkInfo("Angel Stadium", 33.8003, -117.8827, 160, "open", 73.0),
    "OAK": ParkInfo("Oakland Coliseum", 37.7516, -122.2005, 20, "open", 60.0),
    "SEA": ParkInfo("T-Mobile Park", 47.5915, -122.3317, 20, "retractable", 2.0),
    "TEX": ParkInfo("Globe Life Field", 32.7473, -97.0944, 545, "retractable", 135.0),
    "ATL": ParkInfo("Truist Park", 33.8908, -84.4678, 1050, "open", 85.0),
    "MIA": ParkInfo("LoanDepot Park", 25.7780, -80.2195, 8, "retractable", 40.0),
    "NYM": ParkInfo("Citi Field", 40.7571, -73.8458, 12, "open", 45.0),
    "PHI": ParkInfo("Citizens Bank Park", 39.9057, -75.1665, 20, "open", 64.0),
    "WSH": ParkInfo("Nationals Park", 38.8729, -77.0074, 25, "open", 67.0),
    "CHC": ParkInfo("Wrigley Field", 41.9484, -87.6553, 600, "open", 25.0),
    "CIN": ParkInfo("Great American Ball Park", 39.0978, -84.5076, 482, "open", 55.0),
    "MIL": ParkInfo("American Family Field", 43.0280, -87.9711, 635, "retractable", 45.0),
    "PIT": ParkInfo("PNC Park", 40.4469, -80.0057, 730, "open", 72.0),
    "STL": ParkInfo("Busch Stadium", 38.6226, -90.1928, 455, "open", 52.0),
    "ARI": ParkInfo("Chase Field", 33.4455, -112.0667, 1082, "retractable", 30.0),
    "COL": ParkInfo("Coors Field", 39.7559, -104.9942, 5280, "open", 50.0),
    "LAD": ParkInfo("Dodger Stadium", 34.0739, -118.2400, 515, "open", 15.0),
    "SD":  ParkInfo("Petco Park", 32.7076, -117.1570, 17, "open", 24.0),
    "SF":  ParkInfo("Oracle Park", 37.7786, -122.3893, 5, "open", 52.0),
}

ROOF_TYPES: set[str] = {"open", "dome", "retractable"}
