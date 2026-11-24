from rdkit import Chem
s = "COC1=CC(=C(C=C1)NC(=O)CCC2=CC=CC=C2)OC"
m = Chem.MolFromSmiles(s)
print(s)
print(Chem.MolToSmiles(m))
print("atoms", m.GetNumAtoms())
