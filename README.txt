Commande à lancer pour éxécuter correctement le programme :
	On utilise des arguments qui prennent en compte le choix de l'utilisateur lors de l'éxécution du programme


	-----on a choisit une valeur entre 0.04 et 0.15 pour k-----
	-----images valides et existantes dans le même dossier que le programme -----

	/!\ l'ordre des arguments est très important /!\ 

	Exemples de cas qui marchent:

		python3  FINAL_PROJET_ATELIER.py traitement lbp boat.jpg gris 
		python3  FINAL_PROJET_ATELIER.py traitement lbp kh.jpg couleur

		python3  FINAL_PROJET_ATELIER.py traitement poi boat.jpg gris 0.04
		python3  FINAL_PROJET_ATELIER.py traitement poi kh.jpg couleur 0.04
		python3  FINAL_PROJET_ATELIER.py traitement poi kh.jpg couleur 0.08 

		python3  FINAL_PROJET_ATELIER.py traitement blur boat.jpg gris
		python3  FINAL_PROJET_ATELIER.py traitement blur kh.jpg couleur

		python3  FINAL_PROJET_ATELIER.py traitement deblur boat.jpg gris
		python3  FINAL_PROJET_ATELIER.py traitement deblur kh.jpg couleur

		python3  FINAL_PROJET_ATELIER.py automatique routh
		python3  FINAL_PROJET_ATELIER.py automatique jury


	Exemples de cas qui ne marchent pas:

		python3  FINAL_PROJET_ATELIER.py traitement lbp boat.jpg gris fdsfd
		python3  FINAL_PROJET_ATELIER.py traitement sqdsqdsq lbp boat.jpg gris 
		python3  FINAL_PROJET_ATELIER.py gris traitement lbp boat.jpg  
		python3  FINAL_PROJET_ATELIER.py traitement blur boat.jpg
		python3  FINAL_PROJET_ATELIER.py blur boat.jpg automatique
		
