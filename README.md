# Práctica 1. Python básico. Min heaps
### **ÍNDICE**
1. **[Python básico](#python_básico)**

	- [Multiplicación de matrices](#mm)
	- [Busqueda binaria](#bb)
2. **[Min heaps](#min_heaps)**

	- [Actualizar heap](#update_heap)
	- [Crear min heap](#create_heap)
	- [Insertar elemento](#insert_heap)
3. **[Colas de prioridad](#priority_queues)**

    - [Crear una cola de prioridad](#create_pq)
    - [Insertar un elemento](#insert_pq)
    - [Eliminar un elemento](#remove_pq)
    - [Ordenar un heap](#order_heap)
<br>
<br>

<div id='python_básico'></div>

## **<u>Python básico</u>**

El objetivo de esta parte del proyecto es crear diferentes algoritmos, comprobando su tiempo de ejecución en función de los argumentos de entrada (la longitud de las listas/arrays). Los algoritmos que se van a comprobar son:
- [Multiplicación de matrices](#mm)
- [Busqueda binaria](#bb)
	- [BB iterativa](#bb_iter)
	- [BB recursiva](#bb_rec)
<br><br>
<div id='mm'></div>

### · **<u>Multiplicación de matrices</u>**
Este algoritmo se centra en la multiplicación de dos matrices. Para ello, ambas matrices deben ser compatibles, es decir, el número de columnas de la primera matriz debe coincidir con el numero de filas de la segunda. En el caso en el que esto no se cumpla, se devolverá una matriz vacía.

La siguiente es una posible implementación del algoritmo deseado:
```
def matrix_multiplication(m_1: np.ndarray, m_2: np.ndarray)-> np.ndarray:
    rows_m1, columns_m1 = np.shape(m_1)
    rows_m2, columns_m2 = np.shape(m_2)
    if columns_m1 != rows_m2:
        return np.array([])
    
    res = np.empty((rows_m1, columns_m2))
    for i in range(rows_m1):
        for j in range(columns_m2):
            res[i][j] = 0
            for k in range(columns_m1):
                res[i][j] += m_1[i][k] * m_2[k][j]
    return res
```

<div id='bb'></div>

### · **<u>Busqueda binaria</u>**
El algoritmo de Búsqueda binaria se basa en la busqueda de un elemento en una lista de N elementos. Al contrario que otros algoritmos, solo es aplicable a listas que ya esten ordenadas; por ello, no se recorre las listas de forma lineal, sino que, empezando en el medio, va dividiendo la lista para poder ir acotando el dato hasta encontrarlo(["Divide y venceras"](https://es.wikipedia.org/wiki/Algoritmo_divide_y_vencer%C3%A1s)).

Como todos los algoritmos, hay dos posibles formas de implementarlos:

<div id='bb_rec'></div>

- **<u>Busqueda binaria recursiva</u>**

Cada vez que se llame a la función, se comprueba si el elemento del medio de la lista es el que se quiere buscar o no. En el caso de que sí lo sea, se devuelve su índice; si no lo es, se vuelve a llamar a la función cambiando los límites laterales de la lista, buscando así en sublistas en donde debería estar el dato a encontrar. La condición de parada es que el límite superior sea inferior al límite inferior: en este caso, el número no se encuentra en la tabla.

Una posible implementación podría ser la siguiente:
```
def rec_bb(t: list, f: int, l: int, key: int)-> int:
    if len(t) == 0 or f < 0 or l < 0:
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
```
<br>
<div id='bb_iter'></div>

- **<u>Busqueda binaria iterativa</u>**

Las bases son las mismas que en la búsqueda recursiva: se observa si el elemento del medio es o no el que se busca y actua en consecuencia: si lo es, devuelve el índice; en caso contrario, se cambian los límites para acceder a una subtabla. La diferencia con la recursividad es que todo el codigo se encuentra dentro de un bucle, por lo que al cambiar los límites vuelve a ejecutarse el código sin necesidad de llamar a la función. La condición de parada es la misma.

Una posible implementación podría ser:
```
def bb(t: list, f: int, l: int, key: int)-> int:
    if len(t) == 0 or f < 0 or l < 0:
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
```
<br>
<div id='min_heaps'></div>

## **<u>Min heaps</u>**
Los heaps son estructuras de datos cuyo objetivo es almacenar información de forma eficiente.

Hay dos tipos de heaps:

- Min heaps. Los nodos `padre` deben ser menor a los nodos `hijo`
- Max heaps. Los nodos `padre` deben ser mayor a los nodos `hijo`

![Ejemplo de min heap y max heap](/images/min-max-heaps.png)

En esta parte solo se va a hablar sobre los min heaps, aunque vale también para los max heaps solo que cambiando unos determinados signos. Para ello, se va a explicar como [actualizar un heap](#update_heap), [crear un heap](#create_heap) e [insertar un heap](#insert_heap).

	Nota:

		- Es importante explicar como actualizar un heap antes que como crearlo ya que, para crearlo, es necesario saber cómo saber si un heap está bien formado

		- Algunas funciones importantes como cómo eliminar un elemento de un heap no se explican en esta sección ya que se piden implementar de forma indirecta en la siguiente
<br>
<div id='update_heap'></div>

### · **<u>Actualizar un heap</u>**
Actualizar un heap es sinónimo de comprobar que las relaciones entre los nodos son las correctas, es decir, comprobar que el valor de los nodos hijos es mayor al de los padres. Por ello, la función `heapify` se encarga de comprobar que, desde un índice determinado, el heap está correctamente hecho; en caso contrario, se encarga de cambiar los valores (de forma ascendiente, es decir, desde un nodo hasta el principio de la array/heap) necesarios para que sea así.

Una posible implementación es la siguiente:
```
def min_heapify(h: np.ndarray, i: int):
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
        if  head != i:
            h[i], h[head] = h[head], h[i]
            i = head
        else:
            flag = False
```
    Nota: en el caso de los max heaps, solo habría que cambiar las comprobaciones a h[i] < h[left] y h[i] < h[right]
<br>
<div id='create_heap'></div>

### · **<u>Crear un heap</u>**
Como hemos visto, un heap es aquel en el que las relaciones se mntienen bajo cualquier circunstancia. Por ello, podemos recorrer toda la array aplicando la función `heapify` a todos los nodos. Así, estamos seguros que todos los nodos con mayores a sus padres.
```
def create_min_heap(h: np.ndarray):
    if h is None:
        return 
    
    index = (len(h) - 2) // 2

    # Vamos mirando todos los nodos, viendo que se cumpla la relacion padre < hijo
    while index >= 0:
        min_heapify(h, index)
        index -= 1
```
    Nota: al depender de la función 'heapify', está función no varía para los max heaps
<br>
<div id='insert_heap'></div>

### · **<u>Insertar un elemento en un heap</u>**
El procedimiento para insertar un nodo en una min heap es el siguiente:
1. Insertamos el nodo en la ultimos posición posible.
2. Desde ese nodo, miramos si su valor es mayor al de su padre.

    · Si el padre es menor al hijo, se termina la función

    · Si el padre es mayor, se intercambian los valores y se repite la acción con el nuevo padre hasta que se mantengan las relaciones correctamente o hasta que se llegue hasta el principio del heap

Una posible implementación sin necesidad de funciones auxiliares podría ser la siguiente:
```
def insert_min_heap(h: np.ndarray, k: int)-> np.ndarray:
    if h is None or k < 0:
        return None

    # Añadimos el elemento
    h = np.array(list(h) + [k])
    index = len(h) - 1

    # Miramos que el padre sea menor al hijo; sino, intercambiamos y miramos su padre
    while index >= 1 and h[(index - 1) // 2] > h[index]:
        h[(index - 1) // 2], h[index] = h[index], h[(index - 1) // 2]
        index = (index - 1) // 2
    return h
```
No obstante, aprovechando que teníamos hecha la función `heapify`, podemos usarla:
```
def insert_min_heap(h: np.ndarray, k: int)-> np.ndarray:
    if h is None:
        return None

    # Añadimos el elemento
    h = np.array(list(h) + [k])

    # Aplicamos la función heapify
    min_heapify(h, len(h) - 1)

    return h
```
<br>
<div id='priority_queues'></div>

## **<u>Colas de prioridad</u>**
Una cola de prioridad es una estructura de datos en la que, si un elemento tiene mas valor/prioridad que otro, estará colocado delante del resto.

Es posible implementarlo de diversas formas, aunque una de las más comunes es a través de heaps, ya que posee rendimientos más óptimos que otras. Además, como queremos que el primer elemento sea el de mayor prioridad, es decir, menor valor, vamos a usar min heaps.

En esta sección vamos a implementar funciones para [crear una cola de prioridad](#create_pq), [insertar un elemento](#insert_pq) en ella y [eliminar](#remove_pq) el que tenga más prioridad. Además, también podemos [ordenar un heap](#order_heap) de forma más eficiente.<br><br>

<div id='create_pq'>

### · **<u>Crear una cola de prioridad vacia</u>**
Las colas de prioridad las vamos a gestionar como min heaps, que a su vez en todas las funciones las hemos implementado con arrays del módulo numpy; por ello, una cola de prioridad vacía no es mas que una array vacía:
```
def pq_ini():
    return np.array([])
```
<br>
<div id='insert_pq'>

### · **<u>Insertar un elemento</u>**
Como vamos a tratar a las colas de prioridad como min heaps, podemos usar las funciones previamente definidas, lo que nos ahorra tiempo y da un mejor rendimiento, ya que estas funciones las hemos optimizado lo mejor posible. Por ello, insertar un elemento en una cola de prioridad es equivalente a insertar un elemento en un min heap:
```
def pq_insert(h: np.ndarray, k: int)-> np.ndarray:
    if h is None:
        return np.array([])
    return insert_min_heap(h, k)
```
<br>
<div id='remove_pq'>

### · **<u>Eliminar un elemento</u>**
Para eliminar un nodo de una cola de prioridad o de un heap, se debe seguir el siguiente orden:
1. Guardamos en una variable auxiliar el nodo que vamos a eliminar de la cola
2. Reemplazamos el primer la cabeza de la cola por el último elemento de la misma
3. Aplicamos el procedimiento/función `heapify` hasta que la relación padre - hijo de los elementos se cumpla
4. Por último, devolvemos el elemento que hemos extraido y la nueva cola de prioridad
Una posible implementación de todo ello, usando las funciones de la sección anterior, podría ser:
```
def pq_remove(h: np.ndarray)-> Tuple[int, np.ndarray]:
    if h is None or len(h) == 0:
        return(-1, pq_ini())
    
    # Extraemos elemento
    node = h[0]

    # Reemplazamos el primer por el ultimo elemento y organizamos el min_heap con heapify
    h[0] = h[len(h) - 1]
    h = h[0:(len(h)-1)]
    min_heapify(h, 0)

    # Devolvemos la tupla
    return tuple([node, h])
```
<br>
<div id='order_heap'></div>

### **<u>Ordenar un heap</u>**
Otro procedimiento que podemos hacer gracias a los heaps es ordenar una lista/array de forma más eficiente que un algoritmo cotidiano como el selector sort. <br>
Como los elementos deben seguir una estructura determinada (el padre es menor que los hijos), es más fácil saber si un elemento está bien estructurado o no. Por ello, un algoritmo para ordenar una lista/array implentada con heaps debe seguir los siguientes pasos:
1. Creamos una nueva array donde se van a almacenar los elementos ya ordenados.
2. Mientras el heap tenga algún elemento, extraemos sus elementos, y posteriormente aplicamos el procedimiento `heapify` al heap.
Ademñas, si recordamos, la función `pq_remove` se encarga de hacer la mayor parte del segundo paso:
```
def min_heap_sort(h: np.ndarray)-> np.ndarray:
    if h is None:
        return np.array([])
    
    # Creamos la array donde vamos a copiar todos los elementos
    result = np.array([])

    # Ordenamos los elementos de la array 'h'
    create_min_heap(h)

    # Extraemos la cabeza del heap en cada iteracion, que es el elemento más pequeño, y
    # lo añadimos a la creada anteriormente
    while len(h) > 0:
        node, h = pq_remove(h)
        result = np.array(list(result) + [node])
    
    return result
```