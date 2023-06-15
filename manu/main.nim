import nbnQ
proc main1()=
    let config2 = @[
        @[1,0],
        @[2,1,0]
    ]
    let dummies2=1
    let output2= @[1,2]
    var red1=newRedNBN(dummies2,config2,output2)
    let inputs2= @[
        @[0.0],
        @[1.0]
    ]
    let outputs_exp2= @[
        @[0.0,1],
        @[1.0,0]
    ]
    let epochs=100000
    let max_error=0.000001
    let e =red1.learnQ(epochs,0.9,max_error,inputs2,outputs_exp2,false)
    echo "Luego de aprender"
    echo "Error: ",e
    var i = 0
    for inp in inputs2:
        echo "Esperado"
        #echo outputs_exp2[i][0]
        echo outputs_exp2[i]
        i += 1
        echo "Real"
        echo red1.predict(inp)
proc main2()=
    let config2 = @[
        @[3,2,1,0],
        @[4,3,1,0],
        @[5,4,3],
        @[6,5,4,3]
    ]
    let dummies2=3
    let output2= @[5,6]
    var red1=newRedNBN(dummies2,config2,output2)
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
    let epochs=2
    let max_error=0.000001
    let alfa=0.9
    let mu = 9.6
    let e =red1.learnQ(epochs,alfa,max_error,inputs2,outputs_exp2,false)
    echo "Luego de aprender"
    echo "Error: ",e
    var i = 0
    for inp in inputs2:
        echo "Esperado"
        #echo outputs_exp2[i][0]
        echo outputs_exp2[i]
        i += 1
        echo "Real"
        echo red1.predict(inp)
    echo red1.predict(@[1.0,1,1])

main2()