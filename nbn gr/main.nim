import random
import std/tables
randomize()
type 
    Neuron = ref object
        id: int
        prev: seq[Neuron]
        pesos: seq[float]
        value: float
        bias: float
        # Es para las neuronas de entrada
        dummy: bool
    Red = object
        ns:Table[int,Neuron]
        red: seq[seq[int]]
        #Las filas representan las neuronas de entrada
        #Las columnas representan las neuronas de salida
        matriz: seq[seq[float]]
proc randomVec(n:int):seq[float]=
    var vec:seq[float] = @[]
    for i in countup(0,n-1,1):
        vec.add(rand(1.0))
    vec
proc newNeuron(id:int,prev:seq[Neuron],dummy:bool):Neuron=
    #let vec=randomVec(prev.len)
    Neuron(id:id,prev:prev,bias:rand(1.0),dummy:dummy)
proc randPesos(n:Neuron)=
    let vec=randomVec(n.prev.len)
    n.pesos=vec
proc toString(n:Neuron):string=
    var s = "{ id: " & $n.id 
    if n.dummy:
        s &= ", soy dommy}"
        return s
    s &= ", bias:" & $n.bias
    s &= ", prev: [\n"
    for p in n.prev:
        s &= " " & p.toString() & " "
    s &= "\n]"
    s &= ", pesos: [\n"
    for w in n.pesos:
        s &= " " & $w & " "
    s &= "\n]}"
    s
proc rToString(r:Red):string=
    var s = " [\n"
    for id, n in r.ns.pairs:
        s &= n.toString() & "\n"
    s &= "]"
    s

proc newRed(red:seq[seq[int]]):Red=
    
    var ns= initTable[int,Neuron]()
    for fila in red:
        let id = fila[0]
        var i=0
        var n=newNeuron(id, @[] ,true)
        var prev:seq[Neuron] = @[]
        if fila.len==1:
            ns.add(id,n)
            continue 
        
        for col in fila:
            
            if(i != 0):
                prev.add(ns[col])
            i += 1
        n.prev=prev
        n.dummy=false
        n.randPesos()
        ns.add(id,n)
    var matriz= newSeq[seq[float]](ns.len)
    for i in countup(0,matriz.len-1,1):
        matriz[i]=newSeq[float](ns.len)
    Red(ns:ns,red:red,matriz:matriz)
proc salida(n:Neuron,f:proc(x:float):float)=
    var suma = 0.0
    #echo n.toString()
    for i in countup(0,n.prev.len-1,1):
        suma += n.pesos[i]*n.prev[i].value
    suma = f(suma)
    n.value=suma
proc feed_forward(r:Red,input:Table[int,float],output:seq[int],f:proc(x:float):float):Table[int,float]=
    var res = initTable[int,float]()
    for id,valor in input.pairs:
        r.ns[id].value=valor
    for id,n in r.ns.pairs:
        if n.prev.len > 0:
            n.salida(f)
    
    for o in output:
        res.add(o,r.ns[o].value)
    res
proc train_ebp(n:Neuron,epochs:int,learn_rate:float,patterns:seq[Table[int,float]],ys:seq[Table[int,float]],output:seq[int],max_error=0.1)=
    var error =100.0
    for epochs in countup(1,epochs,1):
        for p in patterns:
            if error<=max_error:
                break
proc play()=
    var r:Red=newRed(@[@[1],@[2],@[3,1,2],@[4,1,2,3]])
    #echo r.rToString() 
    var inp=initTable[int,float]()
    inp.add(1,1.0)
    inp.add(2,0.0)
    let output= @[4]
    proc f(x:float):float=x
    let res=r.feed_forward(inp,output,f)
    for id,valor in res.pairs:
        echo id
        echo valor


play()