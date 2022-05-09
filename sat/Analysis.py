from sat.main import *

class senA(sen):   
    def NDVI(self):
        ndvi = es.normalized_diff(sen.arr_st[7], sen.arr_st[3])
        ep.plot_bands(ndvi, cmap="RdYlGn", cols=1, vmin=-1,
                      vmax=1, figsize=(10, 14), title="NDVI")
        plt.show()

    def SAVI(self):
        L = 0.5

        savi = ((sen.arr_st[7] - sen.arr_st[3]) /
                (sen.arr_st[7] + sen.arr_st[3] + L)) * (1 + L)

        ep.plot_bands(savi, cmap="RdYlGn", cols=1,
                      vmin=-1, vmax=1, figsize=(10, 14),title="SAVI")

        plt.show()

    def VARI(self):
        vari = (sen.arr_st[2] - sen.arr_st[3])/ (sen.arr_st[2] + sen.arr_st[3] - sen.arr_st[1])
        ep.plot_bands(vari, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14),title="VARI")
        plt.show()
    
    def MNDWI(self):
        mndwi = es.normalized_diff(sen.arr_st[2],sen.arr_st[10])
        ep.plot_bands(mndwi, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14),title="MNDWI")
        plt.show()
    
    def NDMI(self):
        ndmi = es.normalized_diff(sen.arr_st[7], sen.arr_st[10])
        ep.plot_bands(ndmi, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14),title="NDMI")
        plt.show()
    
    def CMR(self):
        cmr = np.divide(sen.arr_st[10], sen.arr_st[11])
        ep.plot_bands(cmr, cmap="RdYlGn", cols=1, vmin=-1, vmax=1, figsize=(10, 14),title="CMR")
        plt.show()