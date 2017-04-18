



class Well():
    def __init__(self, API, Name):
        self.data = [0 ,1 ,2 ,3 ,4 ,5]
        self.API = API
        self.Name = Name
        self.x = {'Lat': 38.789292, 'Long': 79.37234}
x = Well('03755555','OWF-TEST')
print(x, x.API, x.Name, x.data)
print(x.x['Lat'],x.x['Long'])
for num in x.data:
    print(num)




