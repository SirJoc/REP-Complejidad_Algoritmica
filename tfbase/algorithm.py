import json
from networkx import algorithms
from networkx.algorithms.shortest_paths import weighted
from numpy import double
import pandas as pd
import math
import networkx as nx
from heapq import heappush as push, heappop as pop
from networkx.algorithms import tree


url = "../data/poblaciones.csv"
data = pd.read_csv(url)

# departamentos = ['AMAZONAS','ANCASH','APURIMAC','AREQUIPA','AYACUCHO','CAJAMARCA','CUSCO',
#  'HUANCAVELICA', 'HUANUCO' ,'ICA' ,'JUNIN', 'LA LIBERTAD','LAMBAYEQUE','LIMA',
#  'LORETO' ,'MADRE DE DIOS' ,'MOQUEGUA' ,'PASCO' ,'PIURA', 'PUNO' ,'SAN MARTIN',
#  'TACNA' ,'TUMBES' ,'UCAYALI']

departamentos = ['APURIMAC','AMAZONAS','ANCASH']

def calcularDistancia(cp1,cp2):
    la1, lo1 = float(cp1['LATITUD']), float(cp1['LONGITUD'])
    la2, lo2 = float(cp2['LATITUD']), float(cp2['LONGITUD'])
    
    # lo1, la1, lo2, la2 = map(math.radians, [lo1, la1, lo2, la2])
    # dlo = lo2 - lo1
    # dla = la2 - la1
    # a = math.sin(dla/2)**2 + math.cos(la1) * math.cos(la2) * math.sin(dlo/2)**2
    # c = 2 * math.asin(math.sqrt(a))
    # r = 6371

    # return round(c * r, 2)

    dist = math.sqrt((lo2-lo1)**2 +(la2-la1)**2)

    return dist




def Area_Estudio(DATA_SET, Nom_Provinci, Nom_Distrito):

    list_Provincia = DATA_SET['PROVINCIA'].unique() #no repetir

    Provincia = dict()

    for i, nom1 in enumerate(list_Provincia):
        Provincia[nom1] = DATA_SET[DATA_SET['PROVINCIA'] == nom1]

    #---#

    list_Distrito = Provincia[Nom_Provinci]['DISTRITO'].unique()

    Distrito = dict()

    for j, nom2 in enumerate(list_Distrito):
        Distrito[nom2] = Provincia[Nom_Provinci][Provincia[Nom_Provinci]['DISTRITO'] == nom2]
    
    CentrosPoblados = Distrito[Nom_Distrito]
    #--#
    return CentrosPoblados

#Zona = Area_Estudio(data, 'UTCUBAMBA', 'LONYA GRANDE')

def Gra(cp1, cp2, x):
  G = nx.Graph()
  indice = cp1.index[0] 
  for i, cp1 in x.iterrows():
    G.add_node(i-indice)

  for i, cp1 in x.iterrows():
    for j, cp2 in x.iterrows():
        if cp1['CENTRO POBLADO'] != cp2['CENTRO POBLADO']:     
            G.add_edge(i - indice, j - indice, weight = round(calcularDistancia(cp1, cp2), 2))
  
  return G


responsePath = []
def todo(prov,dist, responsePath):
    Zona = Area_Estudio(data, prov, dist)
    G = Gra(Zona,Zona,Zona)
    recorrido = []


    # for n in G.nodes:
    #    G.nodes[n]['visited'] = False
    # def algoritmo(G, aux, inicio, cont):     
    #     if(cont == len(G)):
    #         return recorrido
    #     else:
    #         minimo = 1e9   
    #         G.nodes[aux]['visited'] = True   
    #         recorrido.append(aux+Zona.index[0])
    #         for i in range(0, len(G)):            
    #             if G.nodes[aux] != G.nodes[i] and G.nodes[i]['visited'] == False:
    #                 pdist = G.edges[aux, i]['weight']
    #                 if pdist < minimo:
    #                     minimo = pdist
    #                     aux = i
    #         cont+=1     
    #         algoritmo(G, aux, inicio, cont) 

    # algoritmo(G,0,0,0)


    def _BFS(G, s, recorrido):
        q = [s]
        G.nodes[s]['visited'] = True
        recorrido.append(s+Zona.index[0])
        while q:
            minx = (-1,111111)
            v = q.pop(0)
            for w in G.neighbors(v):
                if G.nodes[w]['visited'] is not True:
                    minx = (w, min(G.edges[v,w]['weight'], minx[1])) if minx[1] > G.edges[v,w]['weight'] else minx
            if minx[0] != -1 and G.nodes[minx[0]]['visited'] != True:
                q.append(minx[0])
                G.nodes[minx[0]]['visited'] = True
                recorrido.append(minx[0]+Zona.index[0])

    def BFS(G, s):
        for n in G.nodes:
            G.nodes[n]['visited'] = False
        _BFS(G,s, recorrido)
        return recorrido  

    BFS(G,0)

    # def min_path(G, c):
    #     recorrido = []
    #     for h in G.neighbors(c):
    #         recorrido.append(G.edges[c, h]['weight'] )
    #     recorrido.sort()
    #     return min(recorrido)

    # def FuerzaBrutaModif(G, s): 
    #     for u in G:
    #         G.nodes[u]['visited'] = False             
    #         G.nodes[u]['path']    = -1                

    #     queue = [s]
    #     recorrido = []
    #     L = nx.Graph(G)
    #     G.nodes[s]['visited'] = True 
    #     while len(queue) > 0 :            
    #         q = queue[0]
    #         for i in G.neighbors(q):
    #             if not G.nodes[i]['visited'] and L.edges[q, i]['weight'] == min_path(L, q):
    #                 G.nodes[i]['visited']  = True
    #                 queue.append(i) 
    #         L.remove_node(queue[0])  
    #         recorrido.append(q+Zona.index[0])
    #         del queue[0]
    #     return recorrido

    # recorrido = FuerzaBrutaModif(G,0)
    

    
    # def _dfs(G, u):
    #     min = math.inf
    #     sig = None
    #     if not G.nodes[u]['visited']:
    #         G.nodes[u]['visited'] = True
    #         for v in G.neighbors(u):
    #             for edge in G.edges(u):
    #                 if v == edge[1] and G.edges[u, v]['weight'] < min and not G.nodes[v]['visited']:
    #                     min = G.edges[u, v]['weight']
    #                     sig = v
    #                     G.nodes[sig]['π'] = u
    #                     _dfs(G, sig)
        
    # def dfs(G, s):
    #     for u in G.nodes:
    #         G.nodes[u]['visited'] = False
    #         G.nodes[u]['π'] = -1
    #     _dfs(G, s)

    
    # dfs(G,0)

    # for v, info in G.nodes.data():
    #     recorrido[v] = int(info['π'])

    # recorrido.remove(recorrido[0])

    # for i in recorrido:
    #     recorrido[i]+=Zona.index[0]

    # print(recorrido)

    for i in recorrido:  
        for j,row in Zona.iterrows():    
            if(j == i):
                responsePath.append({"cp": row["CENTRO POBLADO"],
                             "lat": float(row["LATITUD"]),
                             "lon": float(row["LONGITUD"])})
    return responsePath

def _BFS(G, s, recorrido):
    q = [s]
    G.nodes[s]['visited'] = True
    recorrido.append(s)
    while q:
        minx = (-1,111111)
        v = q.pop(0)
        for w in G.neighbors(v):
            if G.nodes[w]['visited'] is not True:
                minx = (w, min(G.edges[v,w]['weight'], minx[1])) if minx[1] > G.edges[v,w]['weight'] else minx
        if minx[0] != -1 and G.nodes[minx[0]]['visited'] != True:
            q.append(minx[0])
            G.nodes[minx[0]]['visited'] = True
            recorrido.append(minx[0])

def BFS(G, s):
    recorrido = []
    for n in G.nodes:
        G.nodes[n]['visited'] = False
    _BFS(G,s, recorrido)
    return recorrido

def centroideDistrito(prov,dist):
    zonacentroidedis = Area_Estudio(data,prov, dist)
    sumx = 0
    sumy = 0
    for i, row in zonacentroidedis.iterrows():
        sumx+=double(row["LATITUD"])
        sumy+=double(row["LONGITUD"])
    sumx = sumx/len(zonacentroidedis)
    sumy = sumy/len(zonacentroidedis)
    return sumx, sumy

def centroideProvincia(prov):
    distritos = []
    sumx = 0
    sumy = 0
    
    list_Provincia = data['PROVINCIA'].unique()

    Provincia = dict()

    for i, nom1 in enumerate(list_Provincia):
        Provincia[nom1] = data[data['PROVINCIA'] == nom1]

    dists = Provincia[prov]['DISTRITO'].unique()

    for dist in dists:                  
        zonacentroideprov = centroideDistrito(prov, dist)
        distritos.append(zonacentroideprov)
    for i in distritos:
        sumx+=i[0]
        sumy+=i[1]
    
    sumx = sumx/len(distritos)
    sumy = sumy/len(distritos)
    return sumx, sumy

def centroideDepartamento(dept):
    provincias = []
    sumx = 0
    sumy = 0
    dfdpto = data[data["DEPARTAMENTO"] == dept]
    provs = dfdpto["PROVINCIA"].unique()
    for prov in provs:            
        zonacentroidedept = centroideProvincia(prov)
        provincias.append(zonacentroidedept)
    for i in provincias:
        sumx+=i[0]
        sumy+=i[1]
    sumx = sumx/len(provincias)
    sumy = sumy/len(provincias)
    return sumx, sumy

def calcularDistanciaDept(dp1, dp2):
    la1, lo1 = centroideDepartamento(dp1)
    la2, lo2 = centroideDepartamento(dp2)
    dist = math.sqrt((lo2-lo1)**2 +(la2-la1)**2)

    return dist

def calcularDistanciaProv(pr1,pr2):
    la1, lo1 = centroideProvincia(pr1)
    la2, lo2 = centroideProvincia(pr2)
    dist = math.sqrt((lo2-lo1)**2 +(la2-la1)**2)

    return dist

def calcularDistanciaDist(prov, dist1, dist2):
    la1, lo1 = centroideDistrito(prov, dist1)
    la2, lo2 = centroideDistrito(prov, dist2)
    dist = math.sqrt((lo2-lo1)**2 +(la2-la1)**2)

    return dist
    


def ordernarDept(Zona):
    GDept = nx.Graph()    
    for i, dpto1 in enumerate(Zona):
        GDept.add_node(i)
    for i, dpto2 in enumerate(Zona):
        for j, dpto3 in enumerate(Zona):
            if dpto2 != dpto3:          
                GDept.add_edge(i,j, weight =  round(calcularDistanciaDept(dpto2,dpto3),2)) 
    return GDept

def ordernarProv(Zona):
    GDept = nx.Graph()    
    for i, dpto1 in enumerate(Zona):
        GDept.add_node(i)
    for i, dpto2 in enumerate(Zona):
        for j, dpto3 in enumerate(Zona):
            if dpto2 != dpto3:          
                GDept.add_edge(i,j, weight =  round(calcularDistanciaProv(dpto2,dpto3),2))        
    return GDept


def ordernarDist(Zona):
    GDept = nx.Graph()    
    for i, dpto1 in enumerate(Zona):
        GDept.add_node(i)
    for i, dpto2 in enumerate(Zona):
        for j, dpto3 in enumerate(Zona):
            if dpto2 != dpto3:          
                GDept.add_edge(i,j, weight =  round(calcularDistanciaDist(dpto2,dpto3),2))          
    return GDept

# GDept = ordernarZonas(departamentos)
# print(GDept.nodes())

# print(BFS(GDept, 0))





def peru1():     
    responsePath = []
    dptos = data["DEPARTAMENTO"].unique()
    for dpto in dptos:
        dfdpto = data[data["DEPARTAMENTO"] == dpto]
        provs = dfdpto["PROVINCIA"].unique()         
        if(dpto == 'PIURA'):
            for prov in provs:   
                dfprov = dfdpto[dfdpto["PROVINCIA"] == prov]  
                dists = dfprov["DISTRITO"].unique()            
                for dist in dists:  
                    dfdist = dfprov[dfprov["DISTRITO"] == dist]   
                    responsePath = todo(prov,dist, responsePath)
                    
            # for dist in dists:  
            #     dfdist = dfprov[dfprov["DISTRITO"] == dist]   
            #     # responsePath += todo(prov,dist, responsePath)
            #     print(i)
            #     print(dpto,prov,dist)
            #     i+=1
                

                # generar grafo
                # aplicar algoritmo
                # guardar pathdistrito

            # concatenar pathdistrito de toda la provincia
            # guardar pathprovincia

        # concatenar pathprovincia para todo el depto
        # guardar pathdepartamento

    # concatenar pathdepartamento para todo el peru
    # guardar pathperu
    # generar responsePath    
    
    
    
    # for i in recorrido:  
    #     for j,row in Zona.iterrows():    
    #         if(j == i):
    #             responsePath.append({"cp": row["CENTRO POBLADO"],
    #                          "lat": float(row["LATITUD"]),
    #                          "lon": float(row["LONGITUD"])})



    # for i,row in data.iterrows(): ## de pathperu
    #     responsePath.append({"cp": row["CENTRO POBLADO"],
    #                          "lat": float(row["LATITUD"]),
    #                          "lon": float(row["LONGITUD"])})

    return json.dumps(responsePath)


