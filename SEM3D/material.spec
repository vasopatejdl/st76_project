# material.spec for RTM Forward Run

material 0 {
    domain    = solid;
    deftype   = Lambda_Mu;
    spacedef  = file;
    filename0 = "example_la.h5";
    filename1 = "example_mu.h5";
    filename2 = "example_ds.h5";
};

material 1 { copy = 0; };
material 2 { copy = 0; };

# 8 per layer × 3 layers = 24 + 9  bottom PML regions
material 3  { domain = solidpml; copy = 0; };
material 4  { domain = solidpml; copy = 0; };
material 5  { domain = solidpml; copy = 0; };
material 6  { domain = solidpml; copy = 0; };
material 7  { domain = solidpml; copy = 0; };
material 8  { domain = solidpml; copy = 0; };
material 9  { domain = solidpml; copy = 0; };
material 10 { domain = solidpml; copy = 0; };
material 11 { domain = solidpml; copy = 0; };
material 12 { domain = solidpml; copy = 0; };
material 13 { domain = solidpml; copy = 0; };
material 14 { domain = solidpml; copy = 0; };
material 15 { domain = solidpml; copy = 0; };
material 16 { domain = solidpml; copy = 0; };
material 17 { domain = solidpml; copy = 0; };
material 18 { domain = solidpml; copy = 0; };
material 19 { domain = solidpml; copy = 0; };
material 20 { domain = solidpml; copy = 0; };
material 21 { domain = solidpml; copy = 0; };
material 22 { domain = solidpml; copy = 0; };
material 23 { domain = solidpml; copy = 0; };
material 24 { domain = solidpml; copy = 0; };
material 25 { domain = solidpml; copy = 0; };
material 26 { domain = solidpml; copy = 0; };
material 27 { domain = solidpml; copy = 0; };
material 28 { domain = solidpml; copy = 0; };
material 29 { domain = solidpml; copy = 0; };
material 30 { domain = solidpml; copy = 0; };
material 31 { domain = solidpml; copy = 0; };
material 32 { domain = solidpml; copy = 0; };
material 33 { domain = solidpml; copy = 0; };
material 34 { domain = solidpml; copy = 0; };
material 35 { domain = solidpml; copy = 0; };
