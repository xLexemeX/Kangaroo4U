// See ecfast.h.
#include <cstdlib>
#include <cstring>
#include <vector>
#include <thread>
#include "ecfast.h"

static const int LIMBS = 16;                   // 16 windows of 16 bits
static const int TBLN  = 65536;
static EcPoint* GT;                            // GT[l*TBLN + v] = v * 2^(16l) * G

void EcFastInit()
{
    if (GT)
        return;
    GT = (EcPoint*)malloc((size_t)LIMBS * TBLN * sizeof(EcPoint));
    if (!GT)
    {
        printf("ecfast: out of memory for the comb table\r\n");
        exit(1);
    }
    EcInt one;
    one.Set(1);
    EcPoint b = Ec::MultiplyG(one);
    std::vector<EcPoint> bases(LIMBS);
    for (int l = 0; l < LIMBS; l++)
    {
        bases[l] = b;
        for (int i = 0; i < 16; i++)
            b = Ec::DoublePoint(b);
    }
    std::vector<std::thread> th;
    for (int l = 0; l < LIMBS; l++)
        th.emplace_back([l, &bases]() {
            EcPoint* row = GT + (size_t)l * TBLN;
            row[1] = bases[l];
            //the affine add divides by (x2 - x1), so P + P must be a double
            row[2] = Ec::DoublePoint(bases[l]);
            for (int v = 3; v < TBLN; v++)
                row[v] = Ec::AddPoints(row[v - 1], bases[l]);
        });
    for (auto& t : th)
        t.join();
}

bool EcFastMulG(const u64* k, EcPoint& out)
{
    bool first = true;
    for (int l = 0; l < LIMBS; l++)
    {
        u16 v = (u16)((k[l / 4] >> ((l % 4) * 16)) & 0xFFFF);
        if (!v)
            continue;
        EcPoint& p = GT[(size_t)l * TBLN + v];
        out = first ? p : Ec::AddPoints(out, p);
        first = false;
    }
    return !first;
}

bool EcFastRecDist(const u8* rec, u64* k)
{
    memset(k, 0, 4 * sizeof(u64));
    memcpy(k, rec + 9, 22);
    bool neg = (rec[30] & 0x80) != 0;
    if (neg)
    {
        k[2] |= 0xFFFF000000000000ull;          //sign-extend past bit 175
        k[3] = ~0ull;
        for (int i = 0; i < 4; i++)
            k[i] = ~k[i];
        for (int i = 0; i < 4; i++)
            if (++k[i])
                break;
    }
    return neg;
}

bool EcFastVerifyRec(const u8* key3, const u8* rec, const EcPoint& T, u32* x_top)
{
    u64 d[4];
    bool neg = EcFastRecDist(rec, d);
    EcPoint Q;
    if (!EcFastMulG(d, Q))
        return false;                           //a DP at distance zero never happened
    if (rec[31] != 0)                           //not TAME
    {
        EcPoint t = T;
        if (!neg)
            t.y.NegModP();
        Q = Ec::AddPoints(Q, t);
    }
    const u8* x = (const u8*)Q.x.data;
    if (x_top)
        *x_top = ((const u32*)Q.x.data)[7];
    return !memcmp(x, key3, 3) && !memcmp(x + 3, rec, 9);
}

bool EcFastTargetFromHeader(const u8* hdr, bool even_y, EcTarget& out)
{
    out.range = hdr[0];
    out.dp = hdr[1];
    if ((out.range < 32) || (out.range > 180))
        return false;

    memcpy(out.pub.x.data, hdr + 8, 32);
    out.pub.x.data[4] = 0;
    memcpy(out.start.data, hdr + 40, 32);
    out.start.data[4] = 0;
    out.x32.Set(1);
    out.x32.ShiftLeft(out.range - 5);

    out.pub.y = Ec::CalcY(out.pub.x, even_y);

    //T = pub + (x32 - start)*G, exactly as the solver builds PntToSolve
    EcInt delta = out.x32;
    bool sneg = false;
    if (delta.IsLessThanU(out.start))
    {
        delta = out.start;
        delta.Sub(out.x32);
        sneg = true;
    }
    else
        delta.Sub(out.start);

    EcPoint shift;
    if (!EcFastMulG(delta.data, shift))
        return false;
    if (sneg)
        shift.y.NegModP();
    out.T = Ec::AddPoints(out.pub, shift);
    return true;
}
