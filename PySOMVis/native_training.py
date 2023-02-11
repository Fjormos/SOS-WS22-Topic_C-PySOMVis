# Read data from Java SOMToolbox
from SOMToolBox_Parse import SOMToolBox_Parse

# Use any library for training SOM map (e.x. MiniSOM, SOMOClu, SOMpy, PopSOM etc.)
from pysomvis import PySOMVis
from minisom import MiniSom

def create_weight_file(filename, x_dim, y_dim, vec_dim, weights):
    f = open(filename, 'w')

    heading = f'$TYPE som\n$XDIM {x_dim}\n$YDIM {y_dim}\n$VEC_DIM {vec_dim}\n'
    f.write(heading)

    for x in range(x_dim):
        for y in range(y_dim):
            weight_vec = weights[x][y]
            weight_vec_str = ' '.join(map(str, weight_vec)) + f' SOM_MAP_CHAINLINK({x})({y})(0)\n'
            f.write(weight_vec_str)

    f.close()

# chainlink dataset training
chainlink_idata   = SOMToolBox_Parse("datasets\\chainlink\\chainlink.vec").read_weight_file()

chainlink_big_som = MiniSom(100, 60, 3)
chainlink_big_som.train(chainlink_idata['arr'], 10000)

create_weight_file('chainlink_big_som-1000000-iterations.wgt', 10, 10, 3, chainlink_big_som.get_weights())

vis_big_som = PySOMVis(weights=chainlink_big_som._weights, input_data=chainlink_idata['arr'])
vis_big_som._mainview

chainlink_small_som = MiniSom(10, 10, 3)
chainlink_small_som.train(chainlink_idata['arr'], 10000)
create_weight_file('chainlink_small_som.wgt', 10, 10, 3, chainlink_small_som.get_weights())

vis_smal_som = PySOMVis(weights=chainlink_small_som.get_weights(), input_data=chainlink_idata['arr'])
vis_smal_som._mainview


# 10Clusters dataset training
clusters_idata = SOMToolBox_Parse("datasets\\10clusters\\10clusters.vec").read_weight_file()

clusters_big_som = MiniSom(100, 60, 10)
clusters_big_som.train(clusters_idata['arr'], 10000000)

create_weight_file('clusters_big_som-10000000_iterations.wgt', 100, 60, 10, clusters_big_som.get_weights())

vis_clusters_big_som = PySOMVis(weights=clusters_big_som._weights, input_data=clusters_idata['arr'])
vis_clusters_big_som._mainview


clusters_small_som = MiniSom(10, 10, 10)
clusters_small_som.train(clusters_idata['arr'], 10000)

create_weight_file('clusters_small_som.wgt', 10, 10, 10, clusters_small_som.get_weights())

vis_clusters_small_som = PySOMVis(weights=clusters_small_som._weights, input_data=clusters_idata['arr'])
vis_clusters_small_som._mainview
