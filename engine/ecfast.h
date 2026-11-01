// Windowed scalar multiply for bulk verification.
//
// Ec::MultiplyG is double-and-add with an affine add, so it pays a modular
// inverse for every bit: fine for a handful of points, hopeless for millions.
// This trades 84 MB for a 16 x 16-bit comb table and turns one multiply into at
// most 16 additions, which is what makes verifying a whole DP table a ten
// second job instead of an afternoon.
#pragma once

#include "Ec.h"

//Builds the table. Safe to call more than once; the second call does nothing.
void EcFastInit();

//out = k * G for 0 < k < 2^256. Returns false for k == 0.
bool EcFastMulG(const u64* k, EcPoint& out);

//The one definition of "this record holds", shared by the importer and
//dp-verify so they can never drift apart:
//   TAME   x( |d| * G )
//   WILD   x( |d| * G - T )  when d >= 0,   x( |d| * G + T )  when d < 0
//because x(-Q) == x(Q), so the sign of the distance picks the formula and never
//the sign of the answer. key3 is the 3-byte list prefix the table leaves out of
//the record; rec is the 32 stored bytes; T is PntToSolve, unused for tames.
//x_top, when given, receives the top 32 bits of the recomputed x, which is the
//word the DP condition is tested on. The table stores only the low 12 bytes of
//x, so whether a record is a distinguished point at all is not readable from
//the file: it can only be recovered by walking the distance back onto the curve.
bool EcFastVerifyRec(const u8* key3, const u8* rec, const EcPoint& T, u32* x_top = 0);

//|d| of a record's 176-bit signed distance, into k[4]. Returns true if the
//stored value was negative, false if it was zero or positive.
bool EcFastRecDist(const u8* rec, u64* k);

//The search target and the offsets that define it, rebuilt from the 256-byte
//header a table or a DP log carries. T = pub + (x32 - start)*G is the point the
//solver actually walks toward, and it is what a collision's distances resolve
//against; start and x32 turn that answer back into the puzzle's private key.
//
//The header stores only x of the pubkey, so y's parity is lost and the caller
//has to supply it: dp-verify finds it by trying real wild records, a DP log
//carries it in the header. Shared so the two readers cannot drift apart.
struct EcTarget
{
    EcPoint pub;        //the public key, with the parity the caller chose
    EcPoint T;          //PntToSolve
    EcInt   start;      //-start offset
    EcInt   x32;        //1 << (range - 5), the smooth-edge shift
    int     range;
    int     dp;
};

bool EcFastTargetFromHeader(const u8* hdr, bool even_y, EcTarget& out);
