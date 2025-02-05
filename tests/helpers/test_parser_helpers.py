from Bio.Seq import Seq

class MutateReference():

    @classmethod
    def mutate_reference(cls, reference, instrucs):
        seq = list(reference.seq)
        seq = cls._make_substitution(seq, instrucs)
        seq = cls._make_deletion(seq, instrucs)
        seq = cls._make_insertion(seq, instrucs)
        return Seq("".join(seq).replace("-", ""))

    @staticmethod
    def _make_substitution(seq, instrucs):
        for sub in filter(lambda x: x[0] == "s", instrucs):        
            sub = sub.split("_")
            seq[int(sub[1]) - 1] = sub[-1]
        return seq
    
    @staticmethod
    def _make_deletion(seq, instrucs):
        for deletion in filter(lambda x: x[0] == "d", instrucs):
            deletion = deletion.split("_")
            idx1 = int(deletion[1])
            idx2 = idx1 + len(deletion[-1])
            seq[idx1:idx2] = len(deletion[-1])*["-"]
        return seq
    
    @staticmethod
    def _make_insertion(seq, instrucs):
        refcounter = 0
        for insert in filter(lambda x: x[0] == "i", instrucs):
            insert = insert.split("_")
            idx = int(insert[1]) + refcounter - 1
            bases = insert[-1]
            refcounter += len(bases)
            if idx == 0:
                seq = list(idx) + seq
            else:
                seq = seq[:idx + 1] + list(bases) + seq[idx + 1:]
        return seq