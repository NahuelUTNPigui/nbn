import manunbn
let config2 = @[
    @[3,2,1,0],
    @[4,3,1,0],
    @[5,4,3]
]
let inputs2= @[
    @[0.0,0.0,1],
    @[1.0,1.0,0],
    @[1.0,0.0,1],
    @[0.0,1.0,0],
    @[0.0,0,0]
]
let outputs_exp2= @[
    @[0.0,1],
    @[0.0,1],
    @[1.0,0],
    @[1.0,1],
    @[0.0,1]
]
let dummies2=3
let output2= @[4,5]
var red=newRedNBN2(dummies2,config2, output2)
echo "Antes de aprender"

#for inp in inputs2:
#    echo red.predict(inp)
let epochs=100000
let max_error=0.000001
let alfa=0.9
let e =red.learn_gr2(epochs,alfa,max_error,inputs2,outputs_exp2,false)
echo "Luego de aprender"
echo "Error: ",e
var i = 0
for inp in inputs2:
    echo "Esperado"
    #echo outputs_exp2[i][0]
    echo outputs_exp2[i]
    i += 1
    echo "Real"
    echo red.predict(inp)