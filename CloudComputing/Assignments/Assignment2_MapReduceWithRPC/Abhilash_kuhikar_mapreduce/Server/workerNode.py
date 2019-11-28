# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:47:34 2019

@author: Abhilash
"""

'''
set up a server on the received port number and serve some APIs
Client for this mapper will be master node who will poll each master and server node
establish RPC connection with the master node and expose some APIs
'''

from concurrent import futures
import marshal, types
import rpyc
import yaml

with open('config.yml', 'r') as yml:
    config = yaml.load(yml, Loader=yaml.FullLoader)

class WorkerNode(rpyc.Service):
    def __init__(self, path):
        self.path = path + '/'
        
    def on_connect(self, conn):
        host, port = config['KVStore']['host'], int(config['KVStore']['port'])
        rpyc.core.protocol.DEFAULT_CONFIG['sync_request_timeout']=None
        self.__KVStore = rpyc.connect(host, port, config=rpyc.core.protocol.DEFAULT_CONFIG).root
        pass

    def on_disconnect(self, conn):
        pass
    
    def exposed_execute(self, func, input_data, path, mapper = True):
        if not func:
            print('Provide a valid function')
            return None
        code = marshal.loads(func)
        worker_func = types.FunctionType(code, globals(), "worker_func")
        
        if not mapper:#reducer
            inp = []
            for key in input_data:
                value = self.__KVStore.get(key).split()
                inp.append((key,value))
            input_data = inp
        response = worker_func(input_data)

        if mapper:
            for k,v in response:
                self.__KVStore.append(k, v)
            return True
        return response


def spawnWorker(args):
    ip, port = args[0]
    path = args[1]
    #listen on port 50051
    print('Starting mapper Node. Listening on port ' + port)

    from rpyc.utils.server import ThreadedServer
    rpyc.core.protocol.DEFAULT_CONFIG['sync_request_timeout']=None
    t = ThreadedServer(WorkerNode(path), port=int(port), protocol_config=rpyc.core.protocol.DEFAULT_CONFIG)
    try:
        t.start()
        print('Started: ' + port)
    except Exception:
        t.stop()
    print('Node closed')
    