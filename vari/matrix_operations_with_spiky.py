from pybricks.pupdevices import Motor
from pybricks.parameters import Port
from pybricks.tools import wait, StopWatch
from pybricks.hubs import PrimeHub

# functions
class Matrix:
    def __init__(self, vector, nr, nc):
        self.mat = vector
        self.n_rows= nr
        self.n_columns = nc

    def print(self):
        pointer = 0
        for i in range(self.n_rows):
            print(self.mat[pointer:pointer + self.n_columns])
            pointer += self.n_columns

    def get_value(self,i,j):
        if (i > self.n_rows - 1 or j > self.n_columns -1):
            print("Error! Indexes are exceeding matrix's dimensions")
        else:
            return self.mat[self.n_columns*i + j]

    def mult(self,multiplicator):
        mat=[]
        if self.n_columns == multiplicator.n_rows:
            #rows x columns
            for i in range(self.n_rows):
                for j in range(multiplicator.n_columns):
                    prod = 0
                    for k in range(multiplicator.n_rows):
                        prod += self.get_value(i,k)*multiplicator.get_value(k,j)
                    #mat[(multiplicator.n_columns-1)*i + j] = prod
                    mat.append(prod)
            m = Matrix(mat, self.n_rows,multiplicator.n_columns)
            return m
        else:
            print('Errror! The number of columns of A is not equal to the number of rows of B')



vec = [3,1,0,4,4,2,1,3,7]
vec1 =[1,0,3,4,0,0]
#print(vec[6:9])
mat = Matrix(vec,3,3)
mat1 = Matrix(vec1,3,2)
mat2 = mat.mult(mat1)
mat2.print()
hub = PrimeHub()
k = 0
storage = hub.system.storage(0,read=128)
print(storage)
while k != 10:
    for x in mat2.mat:
        hub.display.number(x)
        print(x)
        wait(5000)
    wait(10000)
    k += 1









#def matrix_moltiplication(mat1,mat2,n_rows,n_columns):