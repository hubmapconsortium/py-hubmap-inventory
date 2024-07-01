import hubmapbags
import os

token = os.getenv("TOKEN")
if token is None:
    print("Error: TOKEN environment variable is not set")
    sys.exit(1)

ncores = 12

hubmapbags.utilities.clean()
hubmapbags.reports.daily(token=token, ncores=ncores)
