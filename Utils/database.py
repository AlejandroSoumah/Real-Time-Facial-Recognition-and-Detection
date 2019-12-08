from Utils.fr_utils import img_to_encoding

def Add_to_Database(Person_Name,Person_Image,database,FRmodel):
    database[Person_Name]=img_to_encoding(Person_Image, FRmodel)
    print(Person_Name + " has been added to the dataset" )

def Remove_from_Database(Person_Name,database):
    if database[Person_Name] in database.values():
        del database[Person_Name]
        print(Person_Name + " has been deleted from the dataset")
    else:
        print(Person_Name + " Is not in the dataset")
