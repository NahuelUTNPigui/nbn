import neo
import random
import math
import std/sequtils
randomize()
#Clases
type
    Peso = object
        valor : float
        valido: bool
    RedNBN2 = object
        config :seq[seq[int]]
        padres :seq[seq[int]]
        pesos_inputs:seq[seq[Peso]]
        pesos_deltas:seq[seq[Peso]]
        output_neurons:seq[int]
        dummies:int
        cantidad_pesos:int
        activ:proc(x:float):float
        der_activ:proc(x:float):float
#Funciones auxiliares
proc makeRandomSeqSeq(M,N:int):seq[seq[float]]=
    var seqseq:seq[seq[float]]= @[]
    for i in countup(0,M-1,1):
        var fila:seq[float]= @[]
        for j in countup(0,N-1,1):
            fila.add(rand(1.0))
        seqseq.add(fila)
    seqseq
proc makeRandomSeqPesos(M,N:int):seq[seq[Peso]]=
    var seqPeso:seq[seq[Peso]] = @[]
    for i in countup(0,M-1,1):
        var fila:seq[Peso]= @[]
        for j in countup(0,N-1,1):
            fila.add(Peso(valor:rand(1.0),valido:false))
        seqPeso.add(fila)
    seqPeso
proc setParent(padres: var seq[seq[int]],son:int,big_father:int,config:seq[seq[int]],dummies:int)=
    let linea = config[son-dummies]
    for son_son in linea:
        if son_son != son and son_son >= dummies:
            if not any(padres[son_son-dummies],proc(x:int):bool=x==big_father):
                setParent(padres,son_son,big_father,config,dummies)
    if not any(padres[son-dummies],proc(x:int):bool=x==big_father):    
        padres[son-dummies].add(big_father)
proc isParent(padres:seq[seq[int]],son:int,big_father:int,dummies:int):bool=
    any(padres[son-dummies],proc(x:int):bool=x==big_father)

proc toStringPesosMtx(mtx:seq[seq[Peso]]):string=
    var s = ""
    for f_i in countup(0,mtx.len-1,1):
        
        s &= "\n[ "
        for c_j in countup(0,mtx[0].len-1,1):
            s &= "| "
            if mtx[f_i][c_j].valido: s &= $round(mtx[f_i][c_j].valor,2) else: s &= "cero"
            s &= " |"
        s &= " ]"
    s


proc distributiva(j:var seq[float],n:float)=
    for i in countup(0,j.len-1,1):
        j[i] *= n

proc newRedNBN2(dummies:int,config:seq[seq[int]],output_neurons:seq[int],activ:proc(x:float):float,der_activ:proc(x:float):float):RedNBN2=
    var pesos_inputs=makeRandomSeqPesos(config.len,dummies+1)
    var pesos_deltas=makeRandomSeqPesos(config.len,config.len)
    var padres=newSeq[seq[int]](config.len)
    var j=0
    var cantidad_pesos = 0
    for fila in config:
        padres[j]=newSeq[int]()
        let id_fila=fila[0]-dummies
        for i in countup(1,fila.len-1,1):
            #La ultima columna es el bias y siempre es valido
            pesos_inputs[id_fila][dummies].valido=true
            cantidad_pesos += 1
            if(fila[i]<dummies):
                pesos_inputs[id_fila][fila[i]].valido=true
                cantidad_pesos += 1
            else:
                let id_neurona_inp=fila[i]-dummies
                pesos_deltas[id_neurona_inp][id_fila].valido=true
                cantidad_pesos += 1
        j += 1

    for i in countdown(config.len-1,0,1):
        let linea=config[i]
        for son in linea:
            if son>=dummies and not any(padres[son-dummies],proc(x:int):bool=x==linea[0]):
                setParent(padres,son,linea[0],config,dummies)
    
    RedNBN2(config:config,output_neurons:output_neurons,activ:activ,der_activ:der_activ,dummies:dummies,pesos_inputs:pesos_inputs,pesos_deltas:pesos_deltas,cantidad_pesos:cantidad_pesos,padres:padres)

proc sigmoide(x:float):float=
    1/(1+exp(-x))
proc der_sigmoide(x:float):float=
    sigmoide(x)*(1-sigmoide(x))
proc relu(x:float):float=
    if x>=0:
        x
    else:
        0
proc der_relu(x:float):float=
    if x>=0:
        1
    else:
        0
proc predict(red: RedNBN2,input:seq[float]):seq[float]=
    var salida=newSeq[float](red.pesos_inputs.len)
    for fila_i in countup(0,red.pesos_inputs.len-1,1):
        var suma = red.pesos_inputs[fila_i][red.dummies].valor
        for inp in countup(0,input.len-1,1):
            if(red.pesos_inputs[fila_i][inp].valido):
                suma += red.pesos_inputs[fila_i][inp].valor*input[inp]
        if fila_i!=0:
            for fila_pesos_delta in countup(0,fila_i-1,1):
                if(red.pesos_deltas[fila_pesos_delta][fila_i].valido):
                    suma += salida[fila_pesos_delta]*red.pesos_deltas[fila_pesos_delta][fila_i].valor
        salida[fila_i]=red.activ(suma)
    var output:seq[float]= @[]
    
    for o in red.output_neurons:
        output.add(salida[o-red.dummies])
    output
proc actualizar_gr(red: var RedNBN2,j:seq[float],o_n:int,alfa:float)=
    var pesos_i = 0
    for nn_i in countup(0,red.config.len-1,1):
        if isParent(red.padres,nn_i + red.dummies,o_n,red.dummies):
            for c_inp in countup(0,red.dummies,1):
                if red.pesos_inputs[nn_i][c_inp].valido:
                    red.pesos_inputs[nn_i][c_inp].valor -= j[pesos_i]*alfa
                    pesos_i += 1
            for ny_i in countup(0,nn_i-1,1):
                if red.pesos_deltas[ny_i][nn_i].valido:
                    red.pesos_deltas[ny_i][nn_i].valor -= j[pesos_i]*alfa
                    pesos_i += 1
# Para cada output se mejora la salida
proc learn_gr(red: var RedNBN2, iteraciones:int,alfa:float,max_error:float,inputs:seq[seq[float]],outputs:seq[seq[float]])=
    echo "Aprender"
    var error=0.0
    var prev_error=0.0
    let pesos_validos=red.cantidad_pesos
    for epoch in countup(1,iteraciones,1):
        prev_error=error
        error=0.0
        
        for pattern_i in countup(0,inputs.len-1,1):
            let input=inputs[pattern_i]
            let output=outputs[pattern_i]
            var salida=newSeq[float](red.pesos_inputs.len)
            var derivadas_salida=newSeq[float](red.pesos_inputs.len)
            var output_i=0
            #k
            for fila_i in countup(0,red.pesos_inputs.len-1,1):
                var suma = red.pesos_inputs[fila_i][red.dummies].valor
                for inp in countup(0,input.len-1,1):
                    if(red.pesos_inputs[fila_i][inp].valido):
                        suma += red.pesos_inputs[fila_i][inp].valor*input[inp]
                if fila_i!=0:#Es cualquiera esta conversacion
                    for fila_pesos_delta in countup(0,fila_i-1,1):
                        if(red.pesos_deltas[fila_pesos_delta][fila_i].valido):
                            suma += salida[fila_pesos_delta]*red.pesos_deltas[fila_pesos_delta][fila_i].valor
                salida[fila_i]=red.activ(suma)
                derivadas_salida[fila_i]=red.der_activ(suma)
                if red.output_neurons.contains(fila_i+red.dummies):
                    let diff=salida[fila_i]-output[output_i]
                    error += abs diff
                    derivadas_salida[fila_i] *= diff
                    output_i += 1
                    red.pesos_deltas[fila_i][fila_i] = Peso(valor:derivadas_salida[fila_i],valido:true)
                else:
                    red.pesos_deltas[fila_i][fila_i] = Peso(valor:derivadas_salida[fila_i],valido:true)
                #Hasta aca la salida
                if fila_i != 0 : #Es cualquiera porque complejiza innecesariamente
                    #j
                    for columna_pesos_delta in countdown(fila_i-1,0,1):
                        #Calcular los deltas de la lower triangle
                        if isParent(red.padres , columna_pesos_delta + red.dummies,fila_i + red.dummies,red.dummies):
                            var xkj=0.0 #Aca iria la ecuacion 24
                            for i in countup( columna_pesos_delta,fila_i-1,1):
                                let w=red.pesos_deltas[i][fila_i]
                                let d=red.pesos_deltas[i][columna_pesos_delta]
                                if w.valido and d.valido:
                                    xkj += w.valor * d.valor
                            #Ecuacion 25
                            var dkj=red.pesos_deltas[fila_i][fila_i].valor*xkj
                            
                            red.pesos_deltas[fila_i][columna_pesos_delta]=Peso(valor:dkj,valido:true)
                
            #Calculo el gradiente
            #A pensar
            for o_i in countup(0,output.len-1,1):
                #var derivadas_pesos=newSeq[float]()
                var pesos_i=0
                let o_neurona=red.output_neurons[o_i]
                #Los inputs
                for fila_input_bias in countup(0,red.pesos_inputs.len-1,1):
                    for columna_input_bias in countup(0,red.pesos_inputs[0].len-1,1):
                        if columna_input_bias == red.pesos_inputs[0].len-1:
                            #echo bias
                            #Aca mismo se puede modificar por que ya esta todo calculado
                            red.pesos_inputs[fila_input_bias][columna_input_bias].valor -= alfa * red.pesos_deltas[o_neurona-red.dummies][fila_input_bias].valor
                            #derivadas_pesos.add(alfa * red.pesos_deltas[o_i+red.dummies][fila_input_bias].valor)
                            pesos_i += 1
                        else:
                            if isParent(red.padres,fila_input_bias + red.dummies,o_neurona + red.dummies, red.dummies):
                                #echo hijo
                                #echo red.pesos_deltas[o_neurona][fila_input_bias]
                                #echo input[columna_input_bias]
                                #echo red.pesos_inputs[fila_input_bias][columna_input_bias]
                                red.pesos_inputs[fila_input_bias][columna_input_bias].valor -= alfa * red.pesos_deltas[o_neurona][fila_input_bias].valor * input[columna_input_bias]
                                #derivadas_pesos.add(alfa * input[columna_input_bias] * red.pesos_deltas[o_i+red.dummies][fila_input_bias].valor)
                                pesos_i += 1
                

                #Debo recorrer cada neurona y preguntar si es hija de la salida
                #Si es hija de la salida modifico sus pesos
                #fila es la neurona protegonista
                #columna es la de entrada
                for fila_pesos_delta in countup(0,red.pesos_deltas.len-1,1):
                    if isParent(red.padres,fila_pesos_delta + red.dummies,o_neurona,red.dummies):
                        #Si es hija de la salida
                        for columna_pesos_delta in countup(0,fila_pesos_delta-1,1):
                            if red.pesos_deltas[columna_pesos_delta][fila_pesos_delta].valido:
                                red.pesos_deltas[columna_pesos_delta][fila_pesos_delta].valor -= alfa * salida[columna_pesos_delta] * red.pesos_deltas[o_neurona-red.dummies][fila_pesos_delta].valor
                                pesos_i += 1
        echo "Iteracion: ",epoch
        echo "Error:" ,error
        echo ""
        #[
        if abs(prev_error - error ) < 0.00001 :
            echo "No cambio nada"
            echo error
            break
        ]#
        if(error<=max_error):
            break
proc learn_gr2(red: var RedNBN2, iteraciones:int,alfa:float,max_error:float,inputs:seq[seq[float]],outputs:seq[seq[float]],verbose=false):float=
    var error=0.0
    var prev_error=0.0
    for iter in countup(1, iteraciones,1):
        if verbose:
            echo "Iter: ",iter

        var salida=newSeq[float](red.config.len)
        var der_salida=newSeq[float](red.config.len)
        prev_error=error
        error=0
        
        for p_i in countup(0,inputs.len-1,1):
            
            let inp_exp=inputs[p_i]
            if verbose:
                echo "patter"
                echo inp_exp
            let out_exp=outputs[p_i]
            salida=newSeq[float](red.config.len)
            der_salida=newSeq[float](red.config.len)
            #k
            for nn_i in countup(0,red.config.len-1,1):
                #salida de las neuronas
                var suma= red.pesos_inputs[nn_i][red.dummies].valor
                for inp_i in countup(0,red.dummies-1,1):
                    if red.pesos_inputs[nn_i][inp_i].valido:
                        suma += red.pesos_inputs[nn_i][inp_i].valor * inp_exp[inp_i]
                for nx_i in countup(0,nn_i-1,1):
                    if red.pesos_deltas[nx_i][nn_i].valido:
                        suma += red.pesos_deltas[nx_i][nn_i].valor * salida[nx_i]
                salida[nn_i] = red.activ suma
                der_salida[nn_i] = red.der_activ suma
                #calcular los deltas
                for ny_i in countdown(nn_i,0,1):
                    if ny_i == nn_i:
                        red.pesos_deltas[nn_i][nn_i].valido=true
                        red.pesos_deltas[nn_i][nn_i].valor=der_salida[nn_i]
                    else:
                        if isParent(red.padres,ny_i + red.dummies, nn_i + red.dummies,red.dummies):
                            var xkj=0.0
                            for i in countup(ny_i,nn_i-1,1):
                                if red.pesos_deltas[i][nn_i].valido and red.pesos_deltas[i][ny_i].valido:
                                    xkj += red.pesos_deltas[i][nn_i].valor * red.pesos_deltas[i][ny_i].valor
                            var dkj= xkj * red.pesos_deltas[nn_i][nn_i].valor
                            red.pesos_deltas[nn_i][ny_i].valido=true
                            red.pesos_deltas[nn_i][ny_i].valor=dkj
                #o aca calculo el jacobiano?
            #Aca viene calcular le jacobiano
            for o_i in countup(0, out_exp.len-1,1):
                let o_neurona = red.output_neurons[o_i]
                var j =newSeq[float]()
                
                for nn_i in countup(0,red.config.len-1,1):
                    let derivadas_neurona_out=red.pesos_deltas[o_neurona-red.dummies][nn_i].valor
                    if isParent(red.padres,nn_i + red.dummies,o_neurona,red.dummies):
                        for c_inp in countup(0,red.dummies,1):
                            if red.pesos_inputs[nn_i][c_inp].valido:
                                if c_inp==red.dummies:
                                    j.add(derivadas_neurona_out)
                                else:
                                    j.add(derivadas_neurona_out * inp_exp[c_inp])
                        for nx_i in countup(0,nn_i-1,1):
                            if red.pesos_deltas[nx_i][nn_i].valido:
                                j.add(derivadas_neurona_out * salida[nx_i])
                let diff=salida[o_neurona-red.dummies]-out_exp[o_i]
                if verbose:
                    echo "o_n: ",o_neurona
                    echo "Diff: ",diff
                j.distributiva(diff)
                error += abs(diff)
                if verbose:
                    echo "j"
                    echo j
                if verbose:
                    echo "Pesos deltas antes"
                    echo toStringPesosMtx(red.pesos_deltas)
                    echo ""
                    echo "pesos inputs antes"
                    echo toStringPesosMtx(red.pesos_inputs)
                    echo ""
                actualizar_gr(red,j,o_neurona,alfa)
                if verbose:
                    echo "Pesos deltas despues"
                    echo toStringPesosMtx(red.pesos_deltas)
                    echo ""
                    echo "pesos inputs despues"
                    echo toStringPesosMtx(red.pesos_inputs)
                    echo ""
        if verbose:
            echo "Error: ",error
        if error < max_error or abs(prev_error - error) <= max_error:
            if verbose:
                echo "Iter ",iter
            return error
    error



proc play()=
    #Antes de crear el nbn nahuel
    #Voy a hacer un test hard codeado
    #Con el gradiente, en vez de LM
    let iteraciones=1000
    var mu=1.0
    let beta=0.1
    let alfa=0.9
    let inputs= @[
        @[0.0,0.0],
        @[0.0,1.0],
        @[1.0,0.0],
        @[1.0,1.0],
    ]
    let outputs= @[
        @[0.0],
        @[0.0],
        @[0.0],
        @[1.0],
    ]

    var matriz_inputs_bias=makeRandomSeqSeq(2,3)
    var w12=rand(1.0)
    var delta= rand(1.0)
    var error=2.0
    var error_esperado=0.5
    for iter in countup(1,iteraciones,1):
        echo ""
        echo "Iteracion: ",iter
        var error_sumatoria=0.0
        for fila_input in countup(0,inputs.len-1,1):
            
            #Calculo la salida
            var sumatorias=newSeq[float](2)
            var salidas=newSeq[float](2)
            var slopes=newSeq[float](2)
            var derivadas_neurona=newSeq[float](2)
            for neurona in countup(0,1,1):
                let sumatoria=matriz_inputs_bias[neurona][0] * inputs[fila_input][0] + matriz_inputs_bias[neurona][1] * inputs[fila_input][1] + matriz_inputs_bias[neurona][2]
                sumatorias[neurona]=sumatoria
            salidas[0]=sigmoide(sumatorias[0])
            slopes[0]=der_sigmoide(sumatorias[0])
            sumatorias[1] += salidas[0] * w12
            
            salidas[1]=sigmoide(sumatorias[1])
            slopes[1]=der_sigmoide(sumatorias[1])
            derivadas_neurona[1]=slopes[1]
            
            #Ya tengo la salida
            #Debo recalcular delta
            delta=slopes[1]*w12*slopes[0]
            derivadas_neurona[0]=delta
            #Calculo el error
            let diff=salidas[1]-outputs[fila_input][0]
            error_sumatoria+=abs(diff)
            #Calculo la derivada de los pesos
            var derivadas_error_peso=newSeq[float](7)
            var peso_i=0
            for i in countup(0,1,1):
                for j in countup(0,2,1):
                    if j==2:
                        derivadas_error_peso[peso_i]=derivadas_neurona[i]
                    else:
                        derivadas_error_peso[peso_i]=derivadas_neurona[i] * inputs[fila_input][j]
                    peso_i += 1
            derivadas_error_peso[peso_i]=salidas[0]*derivadas_neurona[1]
            #MArquadt
            #[
            var m=1
            while m<5:
                #Creo el jacobiano
                let jacob_v=makeMatrix(7,1,proc(i,j:int):float=derivadas_error_peso[i])
                let g=jacob_v*diff
                let Q=jacob_v*jacob_v.t
                let H=Q - mu * eye(7)
                let d=solve(H,g)
                #echo "jacob"
                #echo jacob_v
                #echo "H"
                #echo H
                #echo "G"
                #echo g
                #echo "d"
                #echo d
                #Recalculo los pesos apartir de d
                var matriz_inputs_bias_posible:seq[seq[float]]= @[]
                var peso_i_2=0
                for i in countup(0,1,1):
                    var fila:seq[float] = @[]
                    for j in countup(0,2,1):
                        fila.add(matriz_inputs_bias[i][j]-d[peso_i,0])
                        peso_i_2 += 1
                    matriz_inputs_bias_posible.add(fila)
                var w12_posible=w12-d[peso_i_2,0]
                var salida_posible=0.0
                var sumatorias_posibles=newSeq[float](2)
                for neurona in countup(0,1,1):
                    let sumatoria=matriz_inputs_bias_posible[neurona][0] * inputs[fila_input][0] + matriz_inputs_bias_posible[neurona][1] * inputs[fila_input][1] + matriz_inputs_bias_posible[neurona][2]
                    sumatorias_posibles[neurona]=sumatoria
                sumatorias_posibles[1] += sigmoide(sumatorias_posibles[0])*w12     
                salida_posible=sigmoide( sumatorias_posibles[1])
                let diff_posible=salida_posible-outputs[fila_input][0]
                if abs(diff_posible)<abs(diff):
                    mu *= beta
                    #echo "antes"
                    #echo matriz_inputs_bias
                    matriz_inputs_bias=matriz_inputs_bias_posible
                    #echo "despues"
                    #echo matriz_inputs_bias
                    w12=w12_posible
                    break;
                else:
                    #echo "Crece mu"
                    mu /= beta
                    m += 1
                    if m==5:
                        mu=1.0
                    #echo mu
            ]#
            #Steepest descent
            #[
            echo "input"
            echo inputs[fila_input]
            echo "salida esperada"
            echo outputs[fila_input][0]
            echo "Sumatorias"
            echo sumatorias
            echo "salida obtenida"
            echo salidas[1]
            echo "Derivadas neuronas"
            echo derivadas_neurona
            echo "Derivadas pesos"
            echo derivadas_error_peso
            echo "Pesos"
            echo matriz_inputs_bias
            echo w12
            ]#
            var peso_i_2=0
            for i in countup(0,1,1):
                for j in countup(0,2,1):
                    matriz_inputs_bias[i][j] -= alfa*derivadas_error_peso[peso_i_2]
                    peso_i_2 += 1
            w12 -= alfa * derivadas_error_peso[peso_i_2]
        error=error_sumatoria
        echo "Error: ", error
        if error_esperado>=error:
            echo "por error"
            echo error
            break
        if iter==iteraciones:
            echo "Por iteraciones"


    var sumatorias_posibles=newSeq[float](2)
    var salida_posible=0.0
    for fila_input in countup(0,inputs.len-1,1):
        for neurona in countup(0,1,1):
            let sumatoria=matriz_inputs_bias[neurona][0] * inputs[fila_input][0] + matriz_inputs_bias[neurona][1] * inputs[fila_input][1] + matriz_inputs_bias[neurona][2]
            sumatorias_posibles[neurona]=sumatoria
        sumatorias_posibles[1] += sigmoide(sumatorias_posibles[0])*w12     
        salida_posible=sigmoide( sumatorias_posibles[1])
        echo inputs[fila_input]
        echo salida_posible
        echo outputs[fila_input][0]
proc playred()=
    let config3 = @[
    @[2,1,0],
    @[3,1,0],
    @[4,2,3]   
    ]
    let inputs3 = @[
    @[0.0,0.0],
    @[1.0,1.0],
    @[1.0,0.0],
    @[0.0,1.0],
    ]
    let dummies3=2
    let output3= @[3]
    var red=newRedNBN2(dummies3,config3, output3,func(x:float):float=1/(1+exp(-x)),func(x:float):float=(1/(1+exp(-x)))*(1-1/(1+exp(-x))))
    for inp in inputs3:
        echo red.predict(inp)
proc play2()=
    let config1 = @[
        @[2,1,0],
        @[3,2,1,0]
    ]
    let inputs1= @[
        @[0.0,0.0],
        @[1.0,1.0],
        @[1.0,0.0],
        @[0.0,1.0],
    ]
    let outputs_exp1= @[
        @[0.0],
        @[0.0],
        @[1.0],
        @[1.0],
    ]
    let dummies1=2
    let output1= @[3]
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
    var red=newRedNBN2(dummies2,config2, output2,func(x:float):float=1/(1+exp(-x)),func(x:float):float=(1/(1+exp(-x)))*(1-1/(1+exp(-x))))
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
        #echo if red.predict(inp)[0] < 0.5: 0 else: 1
play2()