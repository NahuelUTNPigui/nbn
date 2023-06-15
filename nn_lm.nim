import neo
import random
import math
randomize()
proc showMatrix(m:Matrix[float])=
    for row in m.rows:
        echo row
type
    NN = object
        config:seq[int]
        #Hiper matriz de pesos
        pesos:seq[Matrix[float]]
        #Bias
        biases:seq[Matrix[float]]
proc reLU(x:float):float=
    if x>0:
        x
    else:
        0
proc derReLu(x:float):float=
    if x>0:
        1.0
    else:
        0.0

proc sigmoide(x:float):float=
    1/(1+exp(-x))
proc dersigmoide(x:float):float=
    sigmoide(x)*(1-sigmoide(x))
proc newNN(config:seq[int]):NN=
    var pesos:seq[Matrix[float]] = @[]
    var biases:seq[Matrix[float]] = @[]
    for i in countup(1,config.len-1,1):
        let anterior=config[i-1]
        let actual=config[i]
        var pesos_i=makeMatrix(actual,anterior,proc(i,j:int):float=rand(1.0))
        var bias_i=makeMatrix(actual,1,proc(i,j:int):float=rand(1.0))
        pesos.add(pesos_i)
        biases.add(bias_i)
    NN(config:config,pesos:pesos,biases:biases)
proc predict(nn:NN,input:seq[float]):seq[float]=
    var salida=makeMatrix(input.len,1,proc(i,j:int):float=input[i])
    #echo "input: ",input
    for i in countup(0,nn.pesos.len-1,1):
        let mat=nn.pesos[i]
        let b=nn.biases[i]
        #echo i
        salida=mat*salida+b
        #echo salida
        salida=salida.map(proc(x:float):float=sigmoide(x))
    salida.data
#Por ahora gradiente para demostrar que anda
proc learn_gr(nn:var NN,inputs:seq[seq[float]],outputs:seq[seq[float]],alfa:float,epochs:int,max_error:float,verbose=false,verbose_error=false,verbose_pesos=false):float=
    var epoch=1
    var error=100.0
    var salidas:seq[Matrix[float]] = @[]
    var derivadas:seq[Matrix[float]] =  @[]
    var deltas: seq[Matrix[float]] = @[]
    var derivadas_pesos:seq[Matrix[float]]= @[]
    while error>max_error and epoch < epochs:
        var error_acumulado=0.0
        if verbose or verbose_error or verbose_pesos:
            echo "###################################################"
            echo "EPOCH: ",epoch
            echo "################################################"
        for inp in countup(0,inputs.len-1,1):
            var salida=makeMatrix(inputs[inp].len,1,proc(i,j:int):float=inputs[inp][i])
            var pesos_i = 0
            for p in countup(0,nn.pesos.len-1,1):
                let mat=nn.pesos[p]
                let b=nn.biases[p]
                salida=mat*salida+b
                #[
                if pesos_i != nn.pesos.len-1:
                    let derivada=salida.map(proc(x:float):float=derReLu(x))
                    salida=salida.map(proc(x:float):float=reLU(x))
                    salidas.add(salida)
                    derivadas.add(derivada)
                ]#
                block nombre:
                    
                    let derivada=salida.map(proc(x:float):float=dersigmoide(x))
                    salida=salida.map(proc(x:float):float=sigmoide(x))
                    salidas.add(salida)
                    derivadas.add(derivada)
                pesos_i += 1
            var esperada=makeMatrix(outputs[inp].len,1,proc(i,j:int):float=outputs[inp][i])
            let error_v=  salida - esperada 

            
            if verbose_pesos:
                echo "##################################"
                echo ""
                echo "Pesos antes"
                for p in nn.pesos:
                    echo "fila"
                    p.showMatrix()
                echo ""
                echo "biases antes"
                for b in nn.biases:
                    echo "fila"
                    b.showMatrix()
            if verbose:
                echo "\t####################################"
                echo "\tInputs"
                echo ""
                echo inputs[inp],outputs[inp]
                
                echo ""
                echo "\tSalidas"
                for s in salidas:
                    echo "\t\tfila"
                    s.showMatrix()
                echo ""
                echo "\tDerivadas"
                
                for d in derivadas:
                    echo "\t\tfila"
                    d.showMatrix()
                echo ""
                echo "\tVector error"
                echo error_v
                echo ""
                echo "\tPesos antes"
                for p in nn.pesos:
                    echo "\t\tfila"
                    p.showMatrix()
                echo ""
                echo "\tbiases antes"
                for b in nn.biases:
                    echo "\t\tfila"
                    b.showMatrix()
            let d_0= error_v |*| derivadas[derivadas.len-1]
            deltas.add(d_0)
            
            for p_i in countup(1,nn.pesos.len-1,1):
                let delta = derivadas[derivadas.len-p_i-1] |*| nn.pesos[nn.pesos.len-p_i].t * deltas[p_i-1]
                deltas.add(delta)
            var inp_mat= makeMatrix(inputs[inp].len,1,proc(i,j:int):float=inputs[inp][i])
            derivadas_pesos.add(alfa * deltas[deltas.len-1] * inp_mat.t)
            for p_i in countup(1,nn.pesos.len-1,1):
                derivadas_pesos.add(alfa * deltas[deltas.len-1-p_i] * salidas[p_i-1].t)
            for p_i in countup(0,nn.pesos.len-1,1):
                nn.pesos[p_i]=nn.pesos[p_i] - derivadas_pesos[p_i]
                nn.biases[p_i]=nn.biases[p_i] - alfa * deltas[deltas.len-1-p_i]
            error_acumulado += error_v.l_2()
            salidas= @[]
            derivadas= @[]
            deltas = @[]
            derivadas_pesos= @[]     
        if verbose_error:
            echo "\tError"
            echo error_acumulado
            if abs(error-error_acumulado)>0.01:
                echo "\tHUbo un cambio significativo" 
        error=error_acumulado
        epoch += 1
    error
proc learn_lm(nn:var NN)=
    echo "no implememtado"
proc play()=
    let config = @[2,3,1]
    var nn=newNN(config)
    #echo nn.pesos
    let input= @[1.0,1.0]
    echo nn.predict(input)
proc play2()=
    let m2=makeMatrix(2,3,proc(i,j:int):float=i.toFloat+j.toFloat)
    let m3=m2.map(proc(x:float):float=sigmoide(x))
    echo m2
    echo m3
proc play3()=
    let config = @[2,5,1]
    let inputs = @[
        @[1.0,1.0],
        @[1.0,0.0],
        @[0.0,1.0],
        @[0.0,0.0]
    ]
    let outputs = @[
        @[0.0],
        @[1.0],
        @[1.0],
        @[0.0]
    ]
    let max_error=0.5
    let epochs=5000
    let alfa=0.8
    var nn=newNN(config)

    for i in inputs:
        echo predict(nn,i)
    echo learn_gr(nn,inputs,outputs,alfa,epochs,max_error,false,false,false)
    #echo predict(nn,inputs[0])
    for i in inputs:
        echo predict(nn,i)
proc play4()=
    let config = @[4,5,2]
    let inputs = @[
        @[1.0,1.0,1,0],
        @[1.0,0.0,0,0],
        @[0.0,1.0,1,1],
        @[0.0,0.0,1,0],
    ]
    let outputs = @[
        @[0.0,0.0],
        @[1.0,1],
        @[1.0,0],
        @[0.0,1],
    ]

    let max_error=0.01
    let epochs=5000
    let alfa=0.9
    var nn=newNN(config)


    echo learn_gr(nn,inputs,outputs,alfa,epochs,max_error,false  ,false,false)
    #echo predict(nn,inputs[0])
    proc binary(x:float):float=
        if x>0.5:
            1.0
        else:
            0.0
    for inp in countup(0,inputs.len-1,1):
        echo "real"
        echo outputs[inp]
        echo "Predecir"
        for row in makeMatrix(2,1,proc(i,j:int):float= binary(nn.predict(inputs[inp])[i])).rows():
            echo row
        #echo makeMatrix(2,1,proc(i,j:int):float= binary(nn.predict(inputs[i])[i]))
play3()

