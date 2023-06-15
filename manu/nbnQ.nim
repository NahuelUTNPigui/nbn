import manu
import random
import math
import std/sequtils
randomize()
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
#Clases
type
    Peso = object
        valor : float
        valido: bool
    RedNBN = object
        config :seq[seq[int]]
        padres :seq[seq[int]]
        pesos_inputs:seq[seq[Peso]]
        pesos_deltas:seq[seq[Peso]]
        output_neurons:seq[int]
        dummies:int
        cantidad_pesos:int

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
proc actualizar_pesos(red: var RedNBN,j:seq[float],alfa:float)=
    var pesos_i = 0
    
    for nn_i in countup(0,red.config.len-1,1):
        for c_inp in countup(0,red.dummies,1):
            if red.pesos_inputs[nn_i][c_inp].valido:
                red.pesos_inputs[nn_i][c_inp].valor -= j[pesos_i] * alfa
                pesos_i += 1
        for ny_i in countup(0,nn_i-1,1):
            if red.pesos_deltas[ny_i][nn_i].valido:
                red.pesos_deltas[ny_i][nn_i].valor -= j[pesos_i] * alfa
                pesos_i += 1
proc newRedNBN*(dummies:int,config:seq[seq[int]],output_neurons:seq[int]):RedNBN=
    var pesos_inputs=makeRandomSeqPesos(config.len,dummies+1)
    var pesos_deltas=makeRandomSeqPesos(config.len,config.len)
    var padres=newSeq[seq[int]](config.len)
    var j=0
    var cantidad_pesos = 0
    
    for fila in config:
        padres[j]=newSeq[int]()
        let id_fila=fila[0]-dummies
        pesos_inputs[id_fila][dummies].valido=true
        cantidad_pesos += 1
        for i in countup(1,fila.len-1,1):
            #La ultima columna es el bias y siempre es valido
            
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
    
    RedNBN(config:config,output_neurons:output_neurons,dummies:dummies,pesos_inputs:pesos_inputs,pesos_deltas:pesos_deltas,cantidad_pesos:cantidad_pesos,padres:padres)

proc predict*(red: RedNBN,input:seq[float]):seq[float]=
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
        
        if any(red.output_neurons,proc(x:int):bool = fila_i + red.dummies == x):
            #echo "Es salida: " & $(fila_i + red.dummies)
            salida[fila_i]=sigmoide(suma)
        else:
            salida[fila_i]=relu(suma)
    var output:seq[float]= @[]
    
    for o in red.output_neurons:
        output.add(salida[o-red.dummies])
    output

proc learnQ*(red: var RedNBN, iteraciones:int,alfa:float,max_error:float,inputs:seq[seq[float]],outputs:seq[seq[float]],verbose=false):float=
    var error=0.0
    var prev_error=0.0
    var salida=newSeq[float](red.config.len)
    var der_salida=newSeq[float](red.config.len)
    var cantidad_iteraciones = 0
    for iter in countup(1, iteraciones,1):
        cantidad_iteraciones = iter
        if verbose:
            echo "Iter: ",iter
        prev_error=error
        error=0
        var Q = matrix(red.cantidad_pesos,red.cantidad_pesos,0.0)
        var g = matrix(red.cantidad_pesos,1,0.0)
        for p_i in countup(0,inputs.len-1,1):
            let inp_exp=inputs[p_i]
            if verbose:
                echo "patter"
                echo inp_exp
            let out_exp=outputs[p_i]
            # Diria que no hace falta empezarla de cero
            #salida=newSeq[float](red.config.len)
            #der_salida=newSeq[float](red.config.len)
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
                let o_n = red.output_neurons
                let dm=red.dummies
                if any(o_n,proc(x:int):bool = x == nn_i + dm):
                    salida[nn_i] = sigmoide suma
                    der_salida[nn_i] = der_sigmoide suma
                else:
                    salida[nn_i] = sigmoide suma
                    der_salida[nn_i] = der_sigmoide suma
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
            echo "red.pesos_deltas"
            echo red.pesos_deltas.toStringPesosMtx
            echo "red.inputs"
            echo red.pesos_inputs.toStringPesosMtx
            #Aca viene calcular le jacobiano
            for o_i in countup(0, out_exp.len-1,1):
                let o_neurona = red.output_neurons[o_i]
                var j =newSeq[float](red.cantidad_pesos)
                var j_i=0
                var no_hay_padre=true
                for nn_i in countup(0,red.config.len-1,1):
                    let derivadas_neurona_out=red.pesos_deltas[o_neurona-red.dummies][nn_i].valor
                    if isParent(red.padres,nn_i + red.dummies,o_neurona,red.dummies):
                        no_hay_padre = false
                        for c_inp in countup(0,red.dummies,1):
                            if red.pesos_inputs[nn_i][c_inp].valido:
                                if c_inp==red.dummies:
                                    j[j_i]=derivadas_neurona_out
                                    echo "derivadas_neurona_out: " & $derivadas_neurona_out
                                    j_i += 1
                                else:
                                    j[j_i] = derivadas_neurona_out * inp_exp[c_inp]
                                    echo "derivadas_neurona_out * inp " & $(derivadas_neurona_out * inp_exp[c_inp])
                                    j_i += 1
                        for nx_i in countup(0,nn_i-1,1):
                            if red.pesos_deltas[nx_i][nn_i].valido:
                                j[j_i] = derivadas_neurona_out * salida[nx_i]
                                j_i += 1
                    else:
                        for c_inp in countup(0,red.dummies,1):
                            if red.pesos_inputs[nn_i][c_inp].valido:
                                if c_inp==red.dummies:
                                    j[j_i] = 0.0
                                    j_i += 1
                                else:
                                    j[j_i] = 0.0
                                    j_i += 1
                        for nx_i in countup(0,nn_i-1,1):
                            if red.pesos_deltas[nx_i][nn_i].valido:
                                j[j_i] = 0.0
                                j_i += 1
                let diff=salida[o_neurona-red.dummies]-out_exp[o_i]
                let vector_j = matrix(@[j])
                if no_hay_padre:
                    echo "NO HAY PADRE"
                if verbose:
                    echo "o_n: ",o_neurona
                    echo "Diff: ",diff
                    echo j.len
                let q = vector_j.transpose * vector_j
                let n = vector_j * diff
                echo "q"
                echo q
                echo "diff"
                echo diff
                Q += q
                
                g += n.transpose
                #j.distributiva(diff)
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
        echo Q
        echo g
        let error_pesos = Q.solve(g)
        echo error_pesos
        echo error_pesos.transpose.getArray[0]
        actualizar_pesos(red,error_pesos.transpose.getArray[0],alfa)


    echo "cantidad iteraciones"
    echo cantidad_iteraciones
    error