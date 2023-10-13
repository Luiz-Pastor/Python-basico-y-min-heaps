import numpy as np
from typing import Tuple

def matrix_multiplication(m_1: np.ndarray, m_2: np.ndarray)-> np.ndarray:
    """
    Función que se encarga de multiplicar dos matrices. Para que sea posible,
    la cantidad de columnas de la primera matriz debe coincidir coincidir con
    la cantidad de filas de la segunda
    
    Args: 
        m_1: primera matriz a multiplicar
        m_2: segunda matriz a multiplicar
        
    Returns:
        La matriz resultante del procedimineto de la multiplicación
    """
    if m_1 == None or m_2 == None:
        return np.array([])
    
    dim1 = np.shape(m_1)
    dim2 = np.shape(m_2)
    if dim1[1] != dim2[0]:
        return np.array([])
    res = np.empty((dim1[0], dim2[1]))
    for i in range(dim1[0]):
        for j in range(dim2[1]):
            res[i][j] = 0
            for k in range(dim2[1]):
                res[i][j] += m_1[i][k] * m_2[k][j]
    return res

def bb(t: list, f: int, l: int, key: int)-> int:
    """
    Función que se encarga de encontrar un elemento de una lista empleando 
    el algoritmo de búsqueda binaria de forma iterativa.
    Args: 
        t: la lista en la que vamos a buscar
        f: índice del primer elemento de la lista t
        l: índice del último elemento de la lista t
        key: elemento que queremos encontrar

        
    Returns:
        Si es que hemos encontrado el elemento que buscábamos devolvemos
        el índice de la lista en el que se encuentra el elemento que buscamos(key).
        En caso contrario se devuelve none.
    """
    if t == None or f < 0 or l < 0:
        return None
    while l >= f:
        mid = int((l + f) / 2)
        if t[mid] == key:
            return mid
        elif t[mid] < key:
            f = mid + 1
        else:
            l = mid - 1
    return None

def rec_bb(t: list, f: int, l: int, key: int)-> int:
    """
    Función que se encarga de encontrar un elemento de una lista empleando 
    el algoritmo de búsqueda binaria de forma recursiva.
    Args: 
        t: la lista en la que vamos a buscar
        f: índice del primer elemento de la lista t
        l: índice del último elemento de la lista t
        key: elemento que queremos encontrar

        
    Returns:
        Si es que hemos encontrado el elemento que buscábamos devolvemos
        el índice de la lista en el que se encuentra el elemento que buscamos(key).
        En caso contrario se devuelve none.
    """
    if t == None or f < 0 or l < 0:
        return None
    
    mid = int((l + f) / 2)
    
    if t[mid] == key:
        return mid
    
    if l <= f:
        return None

    elif t[mid] < key:
        return rec_bb(t, mid + 1, l, key)
    
    else:
        return rec_bb(t, f, mid - 1, key)

def min_heapify(h: np.ndarray, i: int):
    """    
    Función que se encarga de ordenar una lista de acuerdo a la estructura de los heaps.
    
    Args: 
        h: array a la cual ejecutar el procedimiento
        i: índice desde el cual empezar
        
    Returns:
        La array con el procedimiento aplicado
    """
    if h == None or i < 0:
        return 

    flag = True  
    while flag:
        head = i
        left = 2*i+1
        right = 2*i+2
        
        if left < len(h) and h[i] > h[left]:
            head = left
        if right < len(h) and h[right] < h[head]:
            head = right
        if  head != i:
            h[i], h[head] = h[head], h[i]
            i = head
        else:
            flag = False

def insert_min_heap(h: np.ndarray, k: int)-> np.ndarray:
    """    
    Función que se encarga de insertar un elemento a una array, 
    de forma que se mantenenga constante la estructura de un heap
    
    Args: 
        h: lista a la cual ejecutar el procedimiento
        k: elemento a añadir
        
    Returns:
        La array con el elemento añadido
    """
    if h == None or k < 0:
        return None

    h = np.array(list(h) + [k])
    index = len(h) - 1
    while index >= 1 and h[(index - 1) // 2] > h[index]:
        h[(index - 1) // 2], h[index] = h[index], h[(index - 1) // 2]
        index = (index - 1) // 2

    return h

def create_min_heap(h: np.ndarray):
    """    
    Función que se encarga crear un min heap a partir de un array de numpy,
    modificando la lista.
    
    Args: 
        h: el array de numpy que vamos a convertir en un heap
       
    Returns:
        None
    """
    if h == None:
        return

    index = (len(h) - 2) // 2

    while index >= 0:
        min_heapify(h, index)
        index -= 1

def pq_ini():
    """    
    Función que se encarga de
    crear una cola de prioridad vacía
    
    Args: 
        None
       
    Returns:
        None
    """
    return np.array([])

def pq_insert(h: np.ndarray, k: int)-> np.ndarray:
    """    
    Función que se encarga de insertar un elemento
    en una cola de prioridad
    
    Args: 
        h: la cola de prioridad en la que se va a insertar
        el elemento
        k: el elemento a insertar
       
    Returns:
        La cola de prioridad con el elemento k ya insertado
    """
    if h == None:
            return None
    return insert_min_heap(h, k)

def pq_remove(h: np.ndarray)-> Tuple[int, np.ndarray]:
    """    
    Función que se encarga de extraer el elemento 
    de mayor prioridad de una cola de prioridad
    
    Args: 
        h: la cola de prioridad en la que se va a extraer
        el elemento
       
    Returns:
        Una tupla cuyo primer elemento es el elemento extraido y 
        cuyo segundo elemento es la cola de prioridad con el 
        elemento ya extraido
    """
    if h == None or len(h) == 0:
        return(-1, pq_ini())
    
    # Extraemos elemento
    node = h[0]

    # Reemplazamos el primer por el ultimo elemento y organizamos el min_heap con heapify
    h[0] = h[len(h) - 1]
    h = h[0:(len(h)-1)]
    min_heapify(h, 0)

    # Devolvemos la tupla
    return tuple([node, h])

def min_heap_sort(h: np.ndarray)-> np.ndarray:
    """    
    Elimina el primer elemento de la array/heap, y lo ordena de acuerdo a la estructura
    de un min heap
    
    Args: 
        h: array a la cual ejecutar el procedimiento
        
    Returns:
        La array con el procedimiento ejecutado
    """
    if h == None:
            return None
    result = np.array([])

    create_min_heap(h)
    while len(h) > 0:
        node, h = pq_remove(h)
        result = np.array(list(result) + [node])
    
    return result