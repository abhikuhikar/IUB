# -*- coding: utf-8 -*-
"""
Created on Tue Nov  5 16:49:19 2019

@author: Abhilash
"""

import marshal
import rpyc
import yaml

def map_fn(text):
    return [(word, '1') for word  in text]

def red_fn(tuples):
    return [(k,len(v)) for k,v in tuples]
    
def main(input_file, output_file, host, port):
    map_fn_ser = marshal.dumps(map_fn.__code__)
    red_fn_ser = marshal.dumps(red_fn.__code__)
    rpyc.core.protocol.DEFAULT_CONFIG['sync_request_timeout']=None
    c = rpyc.connect(host, int(port), config=rpyc.core.protocol.DEFAULT_CONFIG)
    num_clusters = 16
    ip_addresses = [('localhost', str(12345+i)) for i in range(num_clusters)]
    
    cluster_id = c.root.init_cluster(ip_addresses)
    c.root.run_mapred(input_file, map_fn_ser, red_fn_ser, output_file)
    c.root.destroy_cluser(cluster_id)
    
if __name__ == "__main__":
    with open('config.yml', 'r') as yml:
        config = yaml.load(yml, Loader=yaml.FullLoader)
    inputFile, outputFile = config['wordCountClinet']['inputFile'], config['wordCountClinet']['outputfile']
    host, port = config['masterNode']['host'], config['masterNode']['port']
    main(inputFile, outputFile, host, port)
