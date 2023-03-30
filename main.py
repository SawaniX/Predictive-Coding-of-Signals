from read_file import Data
from receiver import Receiver
from transmitter import Transmitter

    
if __name__=='__main__':
    data = Data().data

    transmitter = Transmitter(data).compress()

    receiver = Receiver()
