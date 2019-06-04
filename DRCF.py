import sys;

from keras.layers import Input, Embedding, Flatten, GRU, Dot, Concatenate, Subtract, Activation, SimpleRNN, Multiply, \
    Dense

from keras.models import Model

from keras import backend as K

from BPRGRU import BPRGRU



class DRCF(BPRGRU):

    def __init__(self, dim, maxlen, item_index, negSampleMode="normal"):

        self.dim = dim
        self.maxlen = maxlen
        self.item_index = item_index
        self.dim_mlp = [self.dim * 2, self.dim]
        self.negSampleMode = negSampleMode

        posInput = Input(shape = (1, ))
        negInput = Input(shape = (1, ))
        seqInput = Input(shape = (self.maxlen,))

        itemDotEmbedding = Embedding(len(item_index)+1, self.dim)
        itemMulEmbedding = Embedding(len(item_index)+1, self.dim)
        itemMLPEmbedding = Embedding(len(item_index)+1, self.dim)

        itemSeqDotEmbedding = Embedding(len(item_index)+1, self.dim, mask_zero=True)
        itemSeqMulEmbedding = Embedding(len(item_index)+1, self.dim, mask_zero=True)
        itemSeqMLPEmbedding = Embedding(len(item_index)+1, self.dim, mask_zero=True)

        rnnDot = SimpleRNN(self.dim, unroll=True)
        rnnMul = SimpleRNN(self.dim, unroll=True)
        rnnMLP = SimpleRNN(self.dim, unroll=True)

        uDotEmb = rnnDot(itemSeqDotEmbedding(seqInput))
        pDotEmb = itemDotEmbedding(posInput)
        nDotEmb = itemDotEmbedding(negInput)

        uMulEmb = rnnMul(itemSeqMulEmbedding(seqInput))
        pMulEmb = itemMulEmbedding(posInput)
        nMulEmb = itemMulEmbedding(negInput)

        uMLPEmb = rnnMLP(itemSeqMLPEmbedding(seqInput))
        pMLPEmb = Flatten()(itemMLPEmbedding(posInput))
        nMLPEmb = Flatten()(itemMLPEmbedding(negInput))


        pDot = Dot(axes = -1)([uDotEmb, pDotEmb])
        nDot = Dot(axes = -1)([uDotEmb, nDotEmb])

        pMul = Flatten()(Multiply()([uMulEmb, pMulEmb]))
        nMul = Flatten()(Multiply()([uMulEmb, nMulEmb]))

        pMLP = Concatenate()([uMLPEmb, pMLPEmb])
        nMLP = Concatenate()([uMLPEmb, nMLPEmb])

        for d in self.dim_mlp:
            layer = Dense(d, activation='relu')
            pMLP = layer(pMLP)
            nMLP = layer(nMLP)


        pConcat =  Concatenate()([pDot, pMul, pMLP])
        nConcat =  Concatenate()([nDot, nMul, nMLP])


        finalMLP = Dense(1, activation="linear")


        rel_score = finalMLP(pConcat)
        irr_score = finalMLP(nConcat)

        # Subtract scores.
        diff = Subtract()([rel_score, irr_score])

        # Pass difference through sigmoid function.
        prob = Activation("sigmoid")(diff)


        self.model = Model(inputs = [posInput, negInput, seqInput], outputs = prob)
        self.model.compile(optimizer = "adam", loss = "binary_crossentropy")
        self.get_score = K.function([posInput, seqInput], [rel_score])