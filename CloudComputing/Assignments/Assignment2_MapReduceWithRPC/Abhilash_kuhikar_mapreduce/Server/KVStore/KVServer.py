# -*- coding: utf-8 -*-
"""
Created on Thu Nov  7 19:40:57 2019

@author: Abhilash
"""
import rpyc
import sys
import os
from collections import defaultdict
import yaml

class KVStore(rpyc.Service):
    path = os.getcwd()
    db = path + '/KVStore/data/'
    semaphores = defaultdict(int)

    def on_connect(self, conn):
        pass

    def on_disconnect(self, conn):
        pass
    
    @staticmethod
    def exposed_set(key,value):
        fileName = KVStore.db + key
        
        while KVStore.semaphores[key] == 1:
            continue
        
        KVStore.semaphores[key] = 1
        with open(fileName, 'w') as f:
            f.write(value)
            f.close()
        KVStore.semaphores[key] = 0
    
    @staticmethod
    def exposed_get(key):
        fileName = KVStore.db + key

        while KVStore.semaphores[key] == 1:
            continue
        
        KVStore.semaphores[key] = 1
        value = None
        with open(fileName, 'r') as f:
            value = f.read()
            f.close()
        KVStore.semaphores[key] = 0
        return value
    
    @staticmethod
    def exposed_append(key, value):
        fileName = KVStore.db + key

        while KVStore.semaphores[key] == 1:
            continue
        
        KVStore.semaphores[key] = 1
        with open(fileName, 'a') as f:
            f.write(value + ' ')
            f.close()
        KVStore.semaphores[key] = 0

        return None
    
    @staticmethod
    def exposed_getAllKeys():
        keys = os.listdir(KVStore.db)
        return keys

    @staticmethod
    def exposed_setupPersistentStorage(path):
        os.mkdir(KVStore.db + path)
        KVStore.db += path + '/'
        print('Storage path setup to', KVStore.db)
   
    @staticmethod
    def exposed_removePersistentStorage():
        for file in os.listdir(KVStore.db):
            try:
                os.unlink(os.path.join(KVStore.db,file))
            except Exception as e:
                print(e)
        os.rmdir(KVStore.db)
        print('Storage cleaned for path', KVStore.db)

if __name__ == "__main__":
    with open('config.yml', 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)
    ip, port = config['KVStore']['internal_host'], config['KVStore']['port']
    
    print('Starting KV Store. Listening on port ' + str(port))

    from rpyc.utils.server import ThreadedServer
    rpyc.core.protocol.DEFAULT_CONFIG['sync_request_timeout']=None
    t = ThreadedServer(KVStore, hostname = ip, port=int(port), protocol_config=rpyc.core.protocol.DEFAULT_CONFIG)
    try:
        t.start()
        print('Started: ' + port)
    except Exception:
        t.stop()
    print('KV Store closed')
