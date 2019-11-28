# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 15:48:20 2019

@author: Abhilash
"""
import os
from workerNode import spawnWorker 
import rpyc
import random
import yaml

with open('config.yml', 'r') as yml:
    config = yaml.load(yml, Loader=yaml.FullLoader)

class MapReduceMasterNode(rpyc.Service):

    def __init__(self):
        self.__cluster_id = random.randint(1, 100000000000)
        self.__mappers = []
        self.__reducers = []
        self.__num_workers = 0
        rpyc.core.protocol.DEFAULT_CONFIG['sync_request_timeout']=None
        host, port = config['KVStore']['host'], int(config['KVStore']['port'])
        self.__KVStore = rpyc.connect(host, port, config=rpyc.core.protocol.DEFAULT_CONFIG).root
        self.__KVStore.setupPersistentStorage('cluster_'+str(self.__cluster_id))
    
    def on_connect(self, conn):
        # code that runs when a connection is created
        # (to init the service, if needed)
        self.__worker_connections = []

    def on_disconnect(self, conn):
        # code that runs after the connection has already closed
        # (to finalize the service, if needed)
        pass

    def __spawnWorkers(self):
        for ip_address in self.__ip_addresses:
            pid = os.fork()
            if pid == 0:
                exec(spawnWorker([ip_address, 'cluster_'+str(self.__cluster_id)]))
                break
        print('Workers successfully spawned')

        rpyc.core.protocol.DEFAULT_CONFIG['sync_request_timeout']=None
        for ip, port in self.__ip_addresses:
            self.__worker_connections.append(rpyc.connect(ip, int(port), config=rpyc.core.protocol.DEFAULT_CONFIG).root)
        return True
        
    def exposed_init_cluster(self, ip_addresses, input_type = 'file'):
        self.__ip_addresses = ip_addresses
        self.__num_workers = len(ip_addresses)
        self.cluster_input_type = input_type
        print('init_cluster called', ip_addresses)
        self.__spawnWorkers()
        return self.__cluster_id
    
    def workerInput(self, data, idx):
        n = int(len(data)/self.__num_workers)+1
        return data[n*idx:n*(idx+1)]
        
    def exposed_run_mapred(self, input_data_path, map_fn, red_fn, output_location):
        print('run mapred called')
        self.__map_fn = map_fn
        self.__red_fn = red_fn
        self.__input_data_path = input_data_path

        self.__input_data = None
        if self.cluster_input_type == 'dir':
            self.__input_data = []
            for file in os.listdir(self.__input_data_path):
                if file.endswith(".txt"):
                    self.__input_data.append(os.path.join(self.__input_data_path, file))
        else:
            with open(self.__input_data_path, 'r', encoding = 'utf8') as text_file:
                self.__input_data = text_file.read().lower()
            self.__input_data = self.__input_data.split()

        self.__output_location = output_location

        '''map phase'''
        print('Starting mappers...')

        outPath = 'cluster_'+str(self.__cluster_id)+'/'

        mappers = []
        for i, connection in enumerate(self.__worker_connections):
            mapper_inp = self.workerInput(self.__input_data, i)
            mappers.append(rpyc.async_(connection.execute)(self.__map_fn, mapper_inp, outPath))
            mappers[i].set_expiry(None)
        
        print('Waiting for mappers to complete execution...')
        for mapper in mappers:
            while not mapper.ready:
                pass

        print('Mappers done with execution.')
        '''
        print('Grouping intermediate data....')
        reducer_input = [{} for _ in range(self.__num_workers)]
        for i, mapper in enumerate(mappers):
            intermediate = self.__KVStore.getjson(outPath+'_'+str(i))
            for key, value in intermediate:
                if key not in reducer_input[hash(key)%self.__num_workers]:
                    reducer_input[hash(key)%self.__num_workers][key] = []
                reducer_input[hash(key)%self.__num_workers][key].append(value)
        '''
        print('Grouping intermediate done.')
        '''start reduce phase'''
        print('Starting reducers...')
        self.all_keys = self.__KVStore.getAllKeys()

        reducers = []
        for i, connection in enumerate(self.__worker_connections):
            reducer_inp = self.workerInput(self.all_keys, i)
            reducers.append(rpyc.async_(connection.execute)(self.__red_fn, reducer_inp, outPath, mapper=False))
            reducers[i].set_expiry(None)

        print('Waiting for reducers to complete execution...')
        for reducer in reducers:
            while not reducer.ready:
                pass
        
        '''writing output to file'''
        lines = []
        for i, reducer in enumerate(reducers):
            for k,v in reducer.value:
                lines.append(k + ': ' + str(v))
        
        with open(self.__output_location, 'w') as file:
            file.write('\n'.join(lines))
            

        print('Map Reduce completed.')
        return 'Success'
    
    def exposed_destroy_cluser(self, cluster_id):
        self.__KVStore.removePersistentStorage()
        print('destroy cluster called', cluster_id)
        
    def __isMapTaskCompleted(self):
        pass
    
    def __startReducers(self):
        pass

if __name__ == "__main__":
    from rpyc.utils.server import ThreadedServer
    rpyc.core.protocol.DEFAULT_CONFIG['sync_request_timeout']=None
    ip, master_port = config['masterNode']['internal_host'], config['masterNode']['port']
    t = ThreadedServer(MapReduceMasterNode, hostname = ip, port=int(master_port), protocol_config=rpyc.core.protocol.DEFAULT_CONFIG)
    try:
        print('Starting mapReduce server. Listening on port:', master_port)
        t.start()
    except Exception:
        t.stop()
        
