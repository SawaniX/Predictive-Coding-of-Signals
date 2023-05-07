from read_file import Data
from receiver.receiver import Receiver
from transmitter.transmitter import Transmitter

    
if __name__=='__main__':
    r = 10

    data = Data().data

    transmitter = Transmitter(data, r).compress()

    receiver = Receiver()
