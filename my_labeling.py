__authors__ = ['1633330','1637812','1630920']
__group__ = 'DJ.17'

from utils_data import read_dataset, read_extended_dataset, crop_images, visualize_retrieval, visualize_k_means, Plot3DCloud

from Kmeans import *
from KNN import *
from utils import *
import matplotlib.pyplot as plt

def retrieval_by_color(imatges, labels, pregunta):
    # Funció que rep com a entrada una llista d’imatges, les etiquetes que
    # hem obtingut en aplicar l’algorisme Kmeans a aquestes imatges i la pregunta que fem
    # per a una cerca concreta (això és, un string o una llista d’strings amb els colors que
    # volem buscar). Retorna totes les imatges que contenen les etiquetes de la pregunta que
    # fem. Aquesta funció pot ser millorada afegint un paràmetre d’entrada que contingui el
    # percentatge de cada color que conté la imatge, i retorni les imatges ordenades.

    resultats = []
    indices = []
    info = []

    for i, el in enumerate(imatges):
        if all(color in labels[i] for color in pregunta):
            resultats.append(el)
            indices.append(i)
            info.append(color_labels[i])

    return indices, resultats, info

def retrieval_by_shape(imatges, labels, pregunta):
    # Funció que rep com a entrada una llista d’imatges, les etiquetes que
    # hem obtingut en aplicar l’algorisme KNN a aquestes imatges i la pregunta que fem per
    # a una cerca concreta (això és, un string definint la forma de roba que volem buscar).
    # Retorna totes les imatges que contenen l’etiqueta de la pregunta que fem. Aquesta
    # funció pot ser millorada afegint un paràmetre d’entrada que contingui el percentatge
    # de K-neighbors amb l’etiqueta que busquem i retorni les imatges ordenades.

    images_list = []
    for i in range(len(labels)):
         if labels[i] == pregunta:
             images_list.append(imatges[i])
    return images_list

def retrieval_combined(imatges, shape_labels, color_labels, preguntaF, preguntaC):
    # Funció que rep com a entrada una llista d’imatges, les etiquetes de
    # forma i les de color, una pregunta de forma i una pregunta de color. Retorna les imatges
    # que coincideixen amb les dues preguntes, per exemple: Red Flip Flops. Com en les
    # funcions anteriors, aquesta funció pot ser millorada introduint les dades de percentatge
    # de color i forma de les etiquetes.

    images_list = []
    for i, (col, sh) in enumerate(zip(color_labels, shape_labels)):
        color_yes = False
        shape_yes = False
        for color in col:
            if color in preguntaC:
                color_yes = True
                break

        if preguntaF == sh:
            shape_yes = True

        if color_yes and shape_yes:
            images_list.append(imatges[i])

    return images_list

def Kmean_statistics(kmeans, imagenes, Kmax):
    # Funció que rep com a entrada la classe Kmeans amb un conjunt d’imatges
    # i un valor, Kmax, que representa la màxima K que volem analitzar. Per cada valor des
    # de K=2 fins a K=Kmax executarà la funció fit i calcularà la WCD, el nombre d’iteracions
    # i el temps que ha necessitat per convergir, etc. Finalment, farà una visualització amb
    # aquestes dades.
    pass

def get_shape_accuracy(labels, GT):
    # Funció que rep com a entrada les etiquetes que hem obtingut en
    # aplicar el KNN i el Ground-Truth d’aquestes. Retorna el percentatge d’etiquetes correctes

    total_etiquetes = len(labels)
    labels_correctes = 0

    for i in range(total_etiquetes):
        if labels[i] == GT[i]:
            labels_correctes += 1
    return (labels_correctes / total_etiquetes) * 100

def get_color_accuracy(colors, labels):
    # Funció que rep com a entrada les etiquetes que hem obtingut en
    # aplicar el kmeans i el Ground-Truth d’aquestes. Retorna el percentatge d’etiquetes
    # correctes. Cal tenir en compte que per a cada imatge podem tenir més d’una etiqueta,
    # per tant, heu de pensar com puntuareu si la predicció i el Ground-Truth coincideixen
    # parcialment. A la classe de teoria us varen donar algunes idees per mesurar la similitud
    # entre aquests conjunts.

    total_colors = len(colors)
    colors_correctes = 0
    for i, (col, lab) in enumerate(zip(colors, labels)):
        aux = 0
        aux_num = 0
        for j in range(len(col)):
            if col[j] in lab:
                aux += 1
            aux_num = j
        aux_num += 1
        colors_correctes += aux/(aux_num)
        print(colors_correctes)
    return (colors_correctes / total_colors) * 100

def get_colors(centroids):
    """
    for each row of the numpy matrix 'centroids' returns the color label following the 11 basic colors as a LIST
    Args:
        centroids (numpy array): KxD 1st set of data points (usually centroid points)

    Returns:
        labels: list of K labels corresponding to one of the 11 basic colors
    """

    probs = utils.get_color_prob(centroids) #array con las probabilidades de cada color
    labels = []
    for c in range(len(centroids)):
        max_prob = 0
        i_max_prob = 0
        for i in range(len(probs[c])): #Busqueda de la maxima probabilidad del array
            if probs[c][i] > max_prob:
                max_prob = probs[c][i]
                i_max_prob = i
        labels.append(utils.colors[i_max_prob]) #append del color que corresponde a la maxima probabilidad
    return labels

def menu_principal():
    opciones = {
        '1': ('Test Retrieval_by_color', test_ret_by_color),
        '2': ('Test Retrieval_by_shape', test_ret_by_shape),
        '3': ('Test Retrieval_combined', test_ret_combined),
        '4': ('Show 3DCloud of KMeans', clound_kmeans),
        '5': ('Color accuracy', test_color_accuracy),
        '6': ('Shape accuracy', test_shape_accuracy),
        '7': ('Salir', salir)
    }
    generar_menu(opciones, '7')

def generar_menu(opciones, opcion_salida):
    opcion = None
    while opcion != opcion_salida:
        mostrar_menu(opciones)
        opcion = leer_opcion(opciones)
        ejecutar_opcion(opcion, opciones)
        print()

def mostrar_menu(opciones):
    print('Seleccione una opción:')
    for clave in sorted(opciones):
        print(f' {clave}) {opciones[clave][0]}')

def leer_opcion(opciones):
    while (a := input('Opción: ')) not in opciones:
        print('Opción incorrecta, vuelva a intentarlo.')
    return a

def ejecutar_opcion(opcion, opciones):
    opciones[opcion][1]()

def test_shape_accuracy():
    labels = []
    percents = []
    Ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for i in Ks:
        nn = KNN(train_imgs, train_class_labels)
        knn_list = nn.predict(imgs, i)
        percents.append(get_shape_accuracy(knn_list, class_labels))


    plt.plot(Ks, percents)
    plt.ylim([0, 100])
    plt.xlabel('K')
    plt.ylabel('% of accuracy')
    plt.title("SHAPE ACCURACY")
    plt.show()

def test_color_accuracy():
    colores = []
    percents = []
    Ks = [2, 3, 4, 5, 6, 7, 8, 9, 10]

    for i in Ks:
        for im in imgs[:50]:
            km = KMeans(im, K=i, options={'km_init': 'first'})
            km.fit()
            color = get_colors(km.centroids)
            colores.append(color)

        size = len(colores)
        percents.append(get_color_accuracy(colores, color_labels[:size]))


    plt.plot(Ks, percents)
    plt.ylim([0, 100])
    plt.xlabel('K')
    plt.ylabel('% of accuracy')
    plt.title("COLOR ACCURACY")
    plt.show()


def test_ret_by_color():
    # -------------------------------------------------------------------------------------
    # -----------------------TEST RETRIEVAL_BY_COLOR---------------------------------------
    # -------------------------------------------------------------------------------------
    print(
        ' - Blue \n'
        ' - Red \n'
        ' - Orange \n'
        ' - Brown \n'
        ' - Yellow \n'
        ' - Green \n'
        ' - Purple \n'
        ' - Pink \n'
        ' - Black \n'
        ' - Grey \n'
        ' - White \n'
    )
    a = input('Introduce color: ')
    # Colors to ask
    colors_to_ask = [a]

    colores = []
    #percent = []
    for im in imgs:
        km = KMeans(im, options={'km_init': 'first'})
        km.find_bestK(10)
        color = get_colors(km.centroids)
        colores.append(color)

    # Using retrieval_by_color to apply the question
    indices, matching, info = retrieval_by_color(imgs, colores, colors_to_ask)
    ok_list = [True] * len(colores)
    print(indices)
    print()
    print()
    print()
    print()
    print(color_labels)
    for i in range(len(indices)):
        print(color_labels[indices[i]])
        if colors_to_ask not in color_labels[indices[i]]:
            ok_list[i] = False

    # Visualizing the images that match
    visualize_retrieval(np.array(matching), len(matching), info=info, title='RETRIEVAL_BY_COLOR', ok=ok_list)

def test_ret_by_shape():
    # -------------------------------------------------------------------------------------
    # -----------------------TEST RETRIEVAL_BY_SHAPE---------------------------------------
    # -------------------------------------------------------------------------------------
    print(
        ' - Dresses \n'
        ' - Flip Flops \n'
        ' - Jeans \n'
        ' - Sandals \n'
        ' - Shirts \n'
        ' - Shorts \n'
        ' - Socks \n'
        ' - Handbags \n'
    )
    a = input('Introduce shape: ')
    # Colors to ask
    shapes_to_ask = a
    nn = KNN(train_imgs, train_class_labels)
    knn_list = nn.predict(imgs, 2)
    print(knn_list)
    print()
    print()
    print()
    # Using retrieval_by_color to apply the question
    matching = retrieval_by_shape(imgs, knn_list, shapes_to_ask)
    # Visualizing the images that match
    visualize_retrieval(np.array(matching), len(matching), title='RETRIEVAL_BY_SHAPE')

def test_ret_combined():
    print(
        ' - Dresses \n'
        ' - Flip Flops \n'
        ' - Jeans \n'
        ' - Sandals \n'
        ' - Shirts \n'
        ' - Shorts \n'
        ' - Socks \n'
        ' - Handbags \n'
    )
    a = input('Introduce shape: ')
    print(
        ' - Blue \n'
        ' - Red \n'
        ' - Orange \n'
        ' - Brown \n'
        ' - Yellow \n'
        ' - Green \n'
        ' - Purple \n'
        ' - Pink \n'
        ' - Black \n'
        ' - Grey \n'
        ' - White \n'
    )
    b = input('Introduce color/es: ')
    # Colors and shapes to ask
    shapes_to_ask = a
    colors_to_ask = b
    print(shapes_to_ask)
    print(colors_to_ask)

    colores = []
    for im in imgs:
        km = KMeans(im, options={'km_init': 'first'})
        km.find_bestK(10)
        color = get_colors(km.centroids)
        colores.append(color)

    nn = KNN(train_imgs, train_class_labels)
    knn_list = nn.predict(imgs, 2)
    print(knn_list)
    print()
    print()
    print()
    print()
    print(colores)


    # Using retrieval_by_color to apply the question
    matching = retrieval_combined(imgs, knn_list, colores, shapes_to_ask, colors_to_ask)
    # Visualizing the images that match
    visualize_retrieval(np.array(matching), len(matching), title='RETRIEVAL_COMBINED')

def clound_kmeans():
    # Graficos de las imagenes
    for im in imgs:
        km = KMeans(im)
        km.find_bestK(3)
        Plot3DCloud(km)
        visualize_k_means(km, im.shape) 

def salir():
    print('Saliendo')


if __name__ == '__main__':

    # Load all the images and GT
    train_imgs, train_class_labels, train_color_labels, test_imgs, test_class_labels, \
        test_color_labels = read_dataset(root_folder='./images/', gt_json='./images/gt.json')

    # List with all the existent classes
    classes = list(set(list(train_class_labels) + list(test_class_labels)))

    # Load extended ground truth
    imgs, class_labels, color_labels, upper, lower, background = read_extended_dataset()
    cropped_images = crop_images(imgs, upper, lower)

    menu_principal()
