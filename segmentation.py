import os
import sys
from pathlib import Path
import segmentation_pyx as seg
from utils import Base, engine


if __name__ == '__main__':
    if not os.path.exists('pore_new.db'):
        Base.metadata.create_all(engine)
    base_path = Path(sys.argv[1])
    write_path = Path(sys.argv[2])
    seg.do_all(base_path, write_path)
