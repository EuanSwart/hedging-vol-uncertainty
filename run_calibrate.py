"""fit the band and the prior on 2010-2017 spy data and write data/calibration.json."""
import json

from hedging.calibration import calibrate
from hedging.config import CALIB_FILE

if __name__ == "__main__":
    result = calibrate()
    CALIB_FILE.write_text(json.dumps(result, indent=2))
    print(json.dumps(result, indent=2))
