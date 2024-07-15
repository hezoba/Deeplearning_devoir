1. CONSIDERATION GENERALE

Notre travail a consister à definir une application qui classifie les langues saisie afin de renvoyer la langue correspondante.
Pour mettre en place cette application nous avons utilisé le jeux de donné nommé:"contradictory-my-dear.csv". 

Ce jeux des données contiens les variables "id, premise, hypothesis, lang_abv, language, label". 
Seulement en examinant ce jeux de données nous avons remarqués quelsque incohérence entre le label renvoyé et la langue.
Nous etions alos contraint de réaliser quelque ajustement afin d'utiliser le bon label.

2. MODELE DE CLASSIFICATION DU LANGAGE

Le script du classification de langage s'est réalisé dans le respect des etapes ci'dessous:

-Impotation des bibliothèques
Nous avons importer torch, transformers et pandas

-Chargement du fichier CVS en supprimant les incohérences
Nous avons sauvegardé le fichier "contradictory-my-dear.csv" dans le dossier "classification" de google driver et a partir de colab nous avons appelé ce fichier (cette manoeuvre renforce la performance du GPU)
Nous avons introduit un script de mapping de la variable 'language'qui actualise la variable 'label' qui etait incohérent dans le fichier de base.

-Concatenation des variables 'premise' et 'hypothesis
Conformement à l'enoncé, nous avons concatené les deux variable et former une variable unique qui nous nommons 'text'

-Classe de jeu de donnée
Nous avons definis une classe de jeu de donnée unique

-Initialisation du tokenizer
Ici nous avons utilialiser le tokenizer Bert

-Division des données d'entrainement et test
80% des données sont aloués à l'entrainement et 20% des données au test

-Nous avon ensuite definis le modèle de classification de séquences BERT tout en gerant le CPU afin que l'entrainement soit le plus rapide possible

-Nous avons entrainer le modèle avec la possibilité de varier le nombre d'epoch
-Nous avons sauvegarder le modèle avec la possibilité de le telecharger le fichier "pth" correspendant directement dans notre oridinateur
-Nous avons evaluer le modèle en calculant l'accuracy global.

3. INTERFACE GRADIO ET API DU TEST

Dans un autre fichier nous avons definis un script qui construit l'interface gradio et donner la possibilité de faire un test via un API

-Nous avons commencé par charger les bibliothèques necessaire entre autre torch, transformers, pandas, gradio , sys, et drive
nous avons egalement fait appel à notre premier script "classification_langage",tout en appelant la classe BERTSequenceClassifier

-Nous avions sauvegardé le fichier "pth" de l'entrainement du modèle dans le dossier "classification" dans google driver
Ensuite avons chargé ce fichier "pth" et le fichier "csv"

-Nous avons actualiser à nouveau le label qui etait incohérent

-Nous avons chargé le tokenizer entrainé

-Nous avons definis les interface avec gradio tout en definissant la fonction de prediction de la langue

-Nous vons créer l'interface gradio pour l'API et fianelent lancer cet API


 