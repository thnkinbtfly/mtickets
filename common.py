lang_list = """
en
ceb
de
sv
fr
nl
ru
es
it
pl
ja
vi
war
uk
ar
pt
fa
ca
sr
id
ko
no
fi
tr
hu
cs
ce
sh
ro
tt
eu
ms
he
hy
da
bg
cy
azb
sk
kk
et
min
be
el
hr
lt
gl
az
uz
ur
sl
ka
nn
hi
th
ta
la
mk
ast
bn
lv
tg
af
my
mg
bs
mr
oc
sq
ky
ml
te
sw
br
new
jv
ht
pms
pnb
su
lb
ba
ga
lmo
is
cv
fy
tl
an
sco
pa
io
vo
yo
ne
gu
kn
bar
scn
bpy
mn
nds
zh
"""

def get_list_from_multistring(lang_list, cls=str):
    lang_list = lang_list.split('\n')
    return [cls(x) for x in lang_list if x]

lang_list = get_list_from_multistring(lang_list)


lang2id = {lang: i for i, lang in enumerate(lang_list)}


def sort_dict_by_keys(d):
    return {k: d[k] for k in sorted(d.keys())}
