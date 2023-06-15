import manu
let j = @[@[3.0,1,3]]
let vector_j = matrix(j)
let q = vector_j.transpose * vector_j
echo q
let Q = matrix(3,3,0.0)
echo Q
echo vector_j.getArray[0]