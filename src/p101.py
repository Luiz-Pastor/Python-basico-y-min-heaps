import numpy as np
from typing import Tuple


def matrix_multiplication(m_1: np.ndarray, m_2: np.ndarray) -> np.ndarray:
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
    if m_1 is None or m_2 is None:
        return np.array([])

    # Obtenemos las dimensiones de las matrices y miramos que sean compatibles
    rows_m1, columns_m1 = np.shape(m_1)
    rows_m2, columns_m2 = np.shape(m_2)
    if columns_m1 != rows_m2:
        return np.array([])

    # Creamos una array vacia y vamos metiendo el resultado de las
    # multiplicaciones
    res = np.empty((rows_m1, columns_m2))
    for i in range(rows_m1):
        for j in range(columns_m2):
            res[i][j] = 0
            for k in range(columns_m1):
                res[i][j] += m_1[i][k] * m_2[k][j]
    return res


def bb(t: list, f: int, l: int, key: int) -> int:
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
        el índice de la lista en el que se encuentra el elemento que
        buscamos(key). En caso contrario se devuelve none.
    """
    if (t is None) or (len(t) == 0) or (f < 0) or (l < 0) or (l > len(t)-1):
        return None
    while l >= f:
        mid = int((l + f) / 2)

        # Si el elemento es el del medio, devolvemos su indice
        if t[mid] == key:
            return mid

        # Si el elemento es menor, miramos en la sublista de la derecha
        elif t[mid] < key:
            f = mid + 1

        # Si el elemento es mayor, miramos en la sublista de la izquierda
        else:
            l = mid - 1

    # Si se sale del bucle, no se ha encontardo el elemento
    return None


def rec_bb(t: list, f: int, l: int, key: int) -> int:
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
        el índice de la lista en el que se encuentra el elemento que
        buscamos(key). En caso contrario se devuelve none.
    """
    if (t is None) or (len(t) == 0) or (f < 0) or (l < 0) or (l > len(t)-1):
        return None

    mid = int((l + f) / 2)

    # Si el elemento es el del medio, devolvemos su indice
    if t[mid] == key:
        return mid

    # Si el limite superior es menor al inferior, no se ha encontrado
    # el elemento
    if l <= f:
        return None

    # Si el elemento es menor, miramos en la sublista de la derecha
    elif t[mid] < key:
        return rec_bb(t, mid + 1, l, key)

    # Si el elemento es mayor, miramos en la sublista de la izquierda
    else:
        return rec_bb(t, f, mid - 1, key)


def min_heapify(h: np.ndarray, i: int):
    """
    Función que se encarga de ordenar una lista de acuerdo a la
    estructura de los heaps.

    Args:
        h: array a la cual ejecutar el procedimiento
        i: índice desde el cual empezar

    Returns:
        La array con el procedimiento aplicado
    """
    if h is None or i < 0:
        return

    flag = True
    while flag:
        head = i
        left = 2*i+1
        right = 2*i+2

        # Si el hijo izquierdo es menor al padre, se intercambian
        if left < len(h) and h[i] > h[left]:
            head = left

        # Si el hijo derecho es menor al padre, se intercambian
        if right < len(h) and h[right] < h[head]:
            head = right

        # Si quedan elementos por observar, sigue el bucle
        if head != i:
            h[i], h[head] = h[head], h[i]
            i = head
        else:
            flag = False


def insert_min_heap(h: np.ndarray, k: int) -> np.ndarray:
    """
    Función que se encarga de insertar un elemento a una array,
    de forma que se mantenenga constante la estructura de un heap

    Args:
        h: lista a la cual ejecutar el procedimiento
        k: elemento a añadir

    Returns:
        La array con el elemento añadido
    """
    if h is None:
        return None

    # Añadimos el elemento
    h = np.array(list(h) + [k])

    # Aplicamos heapify desde este último nodo para que se cumplan
    # las relacioens
    min_heapify(h, len(h) - 1)

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
    if h is None:
        return

    index = (len(h) - 2) // 2

    # Vamos mirando todos los nodos, viendo que se cumpla la relacion
    # padre < hijo
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


def pq_insert(h: np.ndarray, k: int) -> np.ndarray:
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
    if h is None:
        return np.array([])
    return insert_min_heap(h, k)


def pq_remove(h: np.ndarray) -> Tuple[int, np.ndarray]:
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
    if h is None or len(h) == 0:
        return (-1, pq_ini())

    # Extraemos elemento
    node = h[0]

    # Reemplazamos el primer por el ultimo elemento y organizamos
    # el min_heap con heapify
    h[0] = h[len(h) - 1]
    h = h[0:(len(h)-1)]
    min_heapify(h, 0)

    # Devolvemos la tupla
    return tuple([node, h])


def min_heap_sort(h: np.ndarray) -> np.ndarray:
    """
    Elimina el primer elemento de la array/heap, y lo ordena de
    acuerdo a la estructura
    de un min heap

    Args:
        h: array a la cual ejecutar el procedimiento

    Returns:
        La array con el procedimiento ejecutado
    """
    if h is None:
        return np.array([])

    # Creamos la array donde vamos a copiar todos los elementos
    result = pq_ini()

    # Ordenamos los elementos de la array 'h'
    create_min_heap(h)

    # Extraemos la cabeza del heap en cada iteracion, que es el elemento
    # más pequeño, y
    # lo añadimos a la creada anteriormente
    while len(h) > 0:
        node, h = pq_remove(h)
        result = np.array(list(result) + [node])

    return result
