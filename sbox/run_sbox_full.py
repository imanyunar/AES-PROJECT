from sbox_generator import generate_sboxes
from sbox_tests import *

sboxes = generate_sboxes()

print("="*60)
print("FULL S-BOX TEST (PAPER COMPLIANT)")
print("="*60)

for name, s in sboxes.items():
    print(f"\nS-box {name}")
    print("Balance :", balance(s))
    print("Biject. :", bijective(s))
    print("NL      :", nonlinearity(s))
    print("SAC     :", round(sac(s), 5))
    print("LAP     :", lap(s))
    print("DAP     :", dap(s))
