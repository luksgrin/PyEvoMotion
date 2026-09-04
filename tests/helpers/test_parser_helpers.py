from Bio.Seq import Seq

class MutateReference():
    """Rebuild a sequence from the reference and its mutation instructions.

    Instruction positions are 1-based reference coordinates for every kind:
    ``s_P_B`` replaces the base at P, ``d_P_BASES`` deletes len(BASES) bases
    starting at P, ``i_P_BASES`` inserts BASES so that the first inserted
    base sits at position P (i.e. before the reference base at P).
    """

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
            idx1 = int(deletion[1]) - 1
            idx2 = idx1 + len(deletion[-1])
            seq[idx1:idx2] = len(deletion[-1])*["-"]
        return seq

    @staticmethod
    def _make_insertion(seq, instrucs):
        # Instructions are sorted by position; each applied insertion shifts
        # the following ones by its length.
        refcounter = 0
        for insert in filter(lambda x: x[0] == "i", instrucs):
            insert = insert.split("_")
            idx = int(insert[1]) - 1 + refcounter
            bases = insert[-1]
            refcounter += len(bases)
            seq = seq[:idx] + list(bases) + seq[idx:]
        return seq
