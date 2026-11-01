// This file is a part of RCKangaroo software
// (c) 2024, RetiredCoder (RC)
// License: GPLv3, see "LICENSE.TXT" file
// https://github.com/RetiredC


#include <iostream>
#include <vector>
#include <thread>

#include "cuda_runtime.h"
#include "cuda.h"

#include "defs.h"
#include "utils.h"
#include "GpuKang.h"
#include "ecfast.h"
#include <signal.h>
#include <time.h>

//64-bit file offsets. MSVC's ftell/fseek take a 32-bit long, which wraps on a
//DP log past 2 GB.
#ifdef _WIN32
	#define k4u_fseek _fseeki64
	#define k4u_ftell _ftelli64
#else
	#define k4u_fseek fseeko
	#define k4u_ftell ftello
#endif


EcJMP EcJumps1[JMP_CNT];
EcJMP EcJumps2[JMP_CNT];
EcJMP EcJumps3[JMP_CNT];

RCGpuKang* GpuKangs[MAX_GPU_CNT];
int GpuCnt;
volatile long ThrCnt;
volatile bool gSolved;

EcInt Int_HalfRange;
EcPoint Pnt_HalfRange;
EcPoint Pnt_NegHalfRange;
EcInt x32;
EcPoint Pntx32;
Ec ec;

CriticalSection csAddPoints;
u8* pPntList;
u8* pPntList2;
volatile int PntIndex;
TFastBase db;
EcPoint gPntToSolve;
EcInt gPrivKey;

volatile u64 TotalOps;
u32 TotalSolved;
u32 gTotalErrors;
volatile u64 PntTotalOps;
bool IsBench;

u32 gDP;
u32 gRange;
EcInt gStart;
bool gStartSet;
EcPoint gPubKey;
u8 gGPUs_Mask[MAX_GPU_CNT];
char gTamesFileName[1024];
double gMax;
bool gGenMode; //tames generation mode
bool gWildOnly; //every kangaroo is a wild (-wildonly)
bool gIsOpsLimit;

//Persistent DP table ("-db"). The DP table IS the banked progress of the search:
//without it every restart discards every distinguished point found so far.
char gDbFileName[1024];
char gDpLogName[1024];
FILE* gDpLogFile;
u64 gDpLogCnt;
u64 gDpLogOps;
int gDbSaveSec = 300;
bool gStoppedByUser;
volatile sig_atomic_t gStopReq;
u64 gPrevOps; //operations banked by earlier runs, carried in the table header
bool gGenTables; //unused placeholder
bool gGenTames; //explicit -gentames, as opposed to RC's -tames/-max flow
#define MAX_IMPORTS 16
char gImportFiles[MAX_IMPORTS][1024];
int gImportCnt;

//table header: identity prefix pins the search, stats live past byte 72


#pragma pack(push, 1)
struct DBRec
{
	u8 x[12];
	u8 d[22];
	u8 type; //0 - tame, 1 - wild1, 2 - wild2
};
#pragma pack(pop)

void InitGpus()
{
	GpuCnt = 0;
	int gcnt = 0;
	cudaGetDeviceCount(&gcnt);
	if (gcnt > MAX_GPU_CNT)
		gcnt = MAX_GPU_CNT;

//	gcnt = 1; //dbg
	if (!gcnt)
		return;

	int drv, rt;
	cudaRuntimeGetVersion(&rt);
	cudaDriverGetVersion(&drv);
	char drvver[100];
	sprintf(drvver, "%d.%d/%d.%d", drv / 1000, (drv % 100) / 10, rt / 1000, (rt % 100) / 10);

	printf("CUDA devices: %d, CUDA driver/runtime: %s\r\n", gcnt, drvver);
	cudaError_t cudaStatus;
	for (int i = 0; i < gcnt; i++)
	{
		cudaStatus = cudaSetDevice(i);
		if (cudaStatus != cudaSuccess)
		{
			printf("cudaSetDevice for gpu %d failed!\r\n", i);
			continue;
		}

		if (!gGPUs_Mask[i])
			continue;

		cudaDeviceProp deviceProp;
		cudaGetDeviceProperties(&deviceProp, i);
		printf("GPU %d: %s, %.2f GB, %d CUs, cap %d.%d, PCI %d, L2 size: %d KB\r\n", i, deviceProp.name, ((float)(deviceProp.totalGlobalMem / (1024 * 1024))) / 1024.0f, deviceProp.multiProcessorCount, deviceProp.major, deviceProp.minor, deviceProp.pciBusID, deviceProp.l2CacheSize / 1024);
		int cm = deviceProp.major * 10 + deviceProp.minor;

		if (deviceProp.major < 6)
		{
			printf("GPU %d - not supported, skip\r\n", i);
			continue;
		}

		cudaSetDeviceFlags(cudaDeviceScheduleBlockingSync);

		GpuKangs[GpuCnt] = new RCGpuKang();
		GpuKangs[GpuCnt]->CudaIndex = i;
		GpuKangs[GpuCnt]->persistingL2CacheMaxSize = deviceProp.persistingL2CacheMaxSize;
		GpuKangs[GpuCnt]->mpCnt = deviceProp.multiProcessorCount;
		GpuKangs[GpuCnt]->JumperInd = GpuCnt;

		if ((cm != 89) && (cm != 120))
		{
			GpuKangs[GpuCnt]->sm_inv_cnt = 0;
			printf("GPU %d: use 3.x version of RCKangaroo to get better performance!\r\n", i);
		}
		else
		{
			GpuKangs[GpuCnt]->Is5xxx = (deviceProp.major == 12);
			GpuKangs[GpuCnt]->sm_inv_cnt = GpuKangs[GpuCnt]->Is5xxx ? (GpuKangs[GpuCnt]->mpCnt / 24) : (GpuKangs[GpuCnt]->mpCnt / 32);
			if (!GpuKangs[GpuCnt]->sm_inv_cnt)
				GpuKangs[GpuCnt]->sm_inv_cnt = 1;
			printf("GPU %d: turbo kernel is enabled!\r\n", i);
		}
		GpuCnt++;
	}
	printf("Total GPUs for work: %d\r\n", GpuCnt);
}
#ifdef _WIN32
u32 __stdcall kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	InterlockedDecrement(&ThrCnt);
	return 0;
}
#else
void* kang_thr_proc(void* data)
{
	RCGpuKang* Kang = (RCGpuKang*)data;
	Kang->Execute();
	__sync_fetch_and_sub(&ThrCnt, 1);
	return 0;
}
#endif
void AddPointsToList(u32* data, int pnt_cnt, u32 KangCnt, u64 ops_cnt, int JumperInd)
{
	for (int i = 0; i < pnt_cnt; i++) //convert KangInd to KangType
	{
		u32* p = data + (GPU_DP_SIZE / 4) * i;
		int KangInd = p[10];
		//Third copy of the split rule, and the only one outside GpuKang.cpp: it
		//must stay identical to TameCntOf() there and to Kparams.TameCnt on the
		//GPU. When these drift, a kangaroo is stamped one type while standing on
		//the other type's point and nothing complains until the records are
		//verified against the curve.
		p[10] = (KangInd < (gWildOnly ? 0 : (int)(KangCnt / 3))) ? TAME : WILD;

		//optional: restart kang after DP
		//GpuKangs[JumperInd]->ToRestartKangaroo(KangInd);
	}
	csAddPoints.Enter();
	if (PntIndex + pnt_cnt >= MAX_CNT_LIST)
	{
		csAddPoints.Leave();
		printf("DPs buffer overflow, some points lost, increase DP value!\r\n");
		return;
	}
	memcpy(pPntList + GPU_DP_SIZE * PntIndex, data, pnt_cnt * GPU_DP_SIZE);
	PntIndex += pnt_cnt;
	PntTotalOps += ops_cnt;
	csAddPoints.Leave();
}

bool Collision_SOTA(EcPoint& pnt, EcInt t, int TameType, EcInt w, int WildType, bool IsNeg)
{
	if (IsNeg)
		t.Neg();
	if (TameType == TAME)
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		EcInt sv = gPrivKey;
		EcPoint P = ec.MultiplyG(gPrivKey);
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		P = ec.MultiplyG(gPrivKey);
		return P.IsEqual(pnt);
	}
	else
	{
		gPrivKey = t;
		gPrivKey.Sub(w);
		if (gPrivKey.data[4] >> 63)
			gPrivKey.Neg();
		gPrivKey.ShiftRight(1);
		EcInt sv = gPrivKey;
		EcPoint P = ec.MultiplyG(gPrivKey);
		if (P.IsEqual(pnt))
			return true;
		gPrivKey = sv;
		gPrivKey.Neg();
		P = ec.MultiplyG(gPrivKey);
		return P.IsEqual(pnt);
	}
}

//One record against the table: insert it, and if it lands on an existing point
//of a type that can close the loop, solve. Shared by the GPU feed and -import.
static bool HandleRec(DBRec& nrec)
{
	DBRec* pref = (DBRec*)db.FindOrAddDataBlock((u8*)&nrec);
	if (gGenMode || !pref)
		return false;

	//in db we dont store first 3 bytes so restore them
	DBRec tmp_pref;
	memcpy(&tmp_pref, &nrec, 3);
	memcpy(((u8*)&tmp_pref) + 3, pref, sizeof(DBRec) - 3);
	pref = &tmp_pref;

	if (pref->type == nrec.type)
	{
		if (pref->type == TAME)
			return false;
		//if it's wild, we can find the key from the same type if distances are different
		if (*(u64*)pref->d == *(u64*)nrec.d)
			return false;
	}

	EcInt w, t;
	int TameType, WildType;
	if (pref->type != TAME)
	{
		memcpy(w.data, pref->d, sizeof(pref->d));
		if (pref->d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
		memcpy(t.data, nrec.d, sizeof(nrec.d));
		if (nrec.d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
		TameType = nrec.type;
		WildType = pref->type;
	}
	else
	{
		memcpy(w.data, nrec.d, sizeof(nrec.d));
		if (nrec.d[21] == 0xFF) memset(((u8*)w.data) + 22, 0xFF, 18);
		memcpy(t.data, pref->d, sizeof(pref->d));
		if (pref->d[21] == 0xFF) memset(((u8*)t.data) + 22, 0xFF, 18);
		TameType = TAME;
		WildType = nrec.type;
	}

	if (!Collision_SOTA(gPntToSolve, t, TameType, w, WildType, false) &&
		!Collision_SOTA(gPntToSolve, t, TameType, w, WildType, true))
	{
		printf("Collision Error\r\n");
		gTotalErrors++;
		return false;
	}
	gSolved = true;
	return true;
}

//A damaged transfer does not read as a short file: it reads as records built
//from misaligned bytes, whose type and distance are nonsense. Merging those
//poisons the table with DPs that can never collide and makes every later import
//log spurious collision errors, so screen them out at the door.
static bool IsPlausibleRec(const DBRec& rec)
{
	if ((rec.type != TAME) && (rec.type != WILD))
		return false;
	//The distance is a signed 176-bit value. A 139-bit range caps its magnitude
	//far below 2^144, so on every real record the top 4 bytes are all 0x00 or
	//all 0xFF. Random bytes clear that test about 1 time in 2^31.
	u8 fill = rec.d[21];
	if ((fill != 0x00) && (fill != 0xFF))
		return false;
	for (int i = 18; i < 21; i++)
		if (rec.d[i] != fill)
			return false;
	return true;
}

//Merge a table produced elsewhere (typically tames from a rented worker) into
//this one. Every record is checked against the curve first: byte plausibility
//only asks whether a record looks like a record, and the failure that actually
//poisoned this table looked perfect. A worker built before the -gentames fix
//started two thirds of its kangaroos at d*G + PntWild and stamped every DP
//TAME, so the file claimed target-independent tames while carrying points
//measured against a target nobody kept. 1.2M of them merged clean and can
//never close a loop. The curve is the only thing that can tell the difference,
//so nothing gets in here without satisfying it.
static bool ImportTable(const char* fn)
{
	FILE* fp = fopen(fn, "rb");
	if (!fp)
	{
		printf("import: cannot open %s\r\n", fn);
		return false;
	}
	u8 hdr[256];
	if (fread(hdr, 1, sizeof(hdr), fp) != sizeof(hdr))
	{
		printf("import: short header\r\n"); fclose(fp); return false;
	}
	if ((hdr[0] != (u8)gRange) || (hdr[1] != (u8)gDP) || (hdr[2] != DBHDR_MARK))
	{
		printf("import: file is range %d dp %d, this search is range %d dp %d\r\n",
			hdr[0], hdr[1], gRange, gDP);
		fclose(fp); return false;
	}
	bool haskey = false;
	for (int i = 0; i < 32; i++)
		if (hdr[8 + i]) haskey = true;
	if (haskey && memcmp(hdr + 8, gPubKey.x.data, 32))
	{
		printf("import: file belongs to a different key\r\n");
		fclose(fp); return false;
	}
	printf("import: merging %s (%s)\r\n", fn, haskey ? "same-key table" : "target-independent tames");

	//A cut-short transfer leaves a valid prefix: keep every record we can read
	//and say how far we got, rather than dropping the file on the first short
	//read and reporting nothing.
	std::vector<DBRec> recs;
	u64 junk = 0, prefixes = 0;
	bool truncated = false;
	for (int i = 0; (i < 256) && !truncated; i++)
		for (int j = 0; (j < 256) && !truncated; j++)
			for (int k = 0; (k < 256) && !truncated; k++)
			{
				u16 cnt;
				if (fread(&cnt, 1, 2, fp) != 2)
				{
					truncated = true;
					break;
				}
				for (int m = 0; m < cnt; m++)
				{
					DBRec rec;
					((u8*)&rec)[0] = (u8)i; ((u8*)&rec)[1] = (u8)j; ((u8*)&rec)[2] = (u8)k;
					if (fread(((u8*)&rec) + 3, 1, 32, fp) != 32)
					{
						truncated = true;
						break;
					}
					if (!IsPlausibleRec(rec))
					{
						junk++;
						continue;
					}
					recs.push_back(rec);
				}
				if (truncated)
					break;
				prefixes++;
			}
	fclose(fp);
	if (truncated)
		printf("import: file is truncated, salvaged %.2f%% of it\r\n",
			100.0 * (double)prefixes / (256.0 * 256.0 * 256.0));
	if (junk)
		printf("import: %llu corrupt records rejected\r\n", (unsigned long long)junk);

	//Verify on the curve. Pure arithmetic per record, so it threads freely; the
	//merge below stays serial because the table does.
	u64 unver = 0;
	if (!recs.empty())
	{
		printf("import: verifying %llu records on the curve...\r\n",
			(unsigned long long)recs.size());
		EcFastInit();
		std::vector<u8> ok(recs.size(), 0);
		//The DP condition is tested on the high word of x, which the file does
		//not store, so a record that is not a distinguished point reads as a
		//perfectly good record until the point is rebuilt. Such a record is not
		//corrupt, it is inert: nothing else will ever stop on that x, so it can
		//never collide. Screen it out here, where the point is already in hand.
		u32 dp_mask = (u32)((1ull << gDP) - 1);
		std::vector<u8> nondp(recs.size(), 0);
		int nth = (int)std::thread::hardware_concurrency();
		if (nth < 1) nth = 4;
		std::vector<std::thread> th;
		for (int t = 0; t < nth; t++)
			th.emplace_back([&, t]() {
				for (size_t r = t; r < recs.size(); r += nth)
				{
					const u8* p = (const u8*)&recs[r];
					u32 top = 0;
					bool v = EcFastVerifyRec(p, p + 3, gPntToSolve, &top);
					if (v && (top & dp_mask)) { nondp[r] = 1; v = false; }
					ok[r] = v ? 1 : 0;
				}
			});
		for (auto& t : th)
			t.join();
		u64 inert = 0;
		for (size_t r = 0; r < recs.size(); r++)
			if (!ok[r])
			{
				unver++;
				if (nondp[r]) inert++;
			}
		unver -= inert;
		if (inert)
			printf("import: %llu records are not dp %d points and could never be "
				"collided with; dropped\r\n", (unsigned long long)inert, gDP);
		if (unver)
			printf("import: %llu records do not sit where their distance says "
				"(%.2f%% of the file); dropped\r\n", (unsigned long long)unver,
				100.0 * (double)unver / (double)recs.size());
		u64 before = db.GetBlockCnt(), total = 0;
		bool solved = false;
		for (size_t r = 0; r < recs.size() && !solved; r++)
		{
			if (!ok[r])
				continue;
			total++;
			if (HandleRec(recs[r]))
				solved = true;
		}
		printf("import: %llu records read, %llu new, table now %llu%s\r\n",
			(unsigned long long)total, (unsigned long long)(db.GetBlockCnt() - before),
			(unsigned long long)db.GetBlockCnt(), solved ? "  *** COLLISION FOUND ***" : "");
	}
	else
		printf("import: nothing usable in the file\r\n");
	return true;
}

void CheckNewPoints()
{
	csAddPoints.Enter();
	if (!PntIndex)
	{
		csAddPoints.Leave();
		return;
	}

	int cnt = PntIndex;
	memcpy(pPntList2, pPntList, GPU_DP_SIZE * cnt);
	PntIndex = 0;
	csAddPoints.Leave();

	for (int i = 0; i < cnt; i++)
	{
		DBRec nrec;
		u8* p = pPntList2 + i * GPU_DP_SIZE;
		memcpy(nrec.x, p, 12);
		memcpy(nrec.d, p + 16, 22);
		nrec.type = gGenMode ? TAME : p[40];
		if (gDpLogFile)
		{
			if (fwrite(&nrec, 1, sizeof(nrec), gDpLogFile) != sizeof(nrec))
			{
				printf("FATAL: DP log write failed, stopping\r\n");
				gStopReq = true;
				break;
			}
			gDpLogCnt++;
			continue;
		}
		if (HandleRec(nrec))
			break;
	}
}

void ShowStats(u64 tm_start, double exp_ops, double dp_val)
{
#ifdef DEBUG_MODE
	for (int i = 0; i <= MD_LEN; i++)
	{
		u64 val = 0;
		for (int j = 0; j < GpuCnt; j++)
		{
			val += GpuKangs[j]->dbg[i];
		}
		if (val)
			printf("Loop size %d: %llu\r\n", i, val);
	}
#endif

	int speed = GpuKangs[0]->GetStatsSpeed();
	for (int i = 1; i < GpuCnt; i++)
		speed += GpuKangs[i]->GetStatsSpeed();

	u64 est_dps_cnt = (u64)(exp_ops / dp_val);
	u64 exp_sec = 0xFFFFFFFFFFFFFFFFull;
	if (speed)
		exp_sec = (u64)((exp_ops / 1000000) / speed); //in sec
	u64 exp_days = exp_sec / (3600 * 24);
	int exp_hours = (int)(exp_sec - exp_days * (3600 * 24)) / 3600;
	int exp_min = (int)(exp_sec - exp_days * (3600 * 24) - exp_hours * 3600) / 60;

	u64 sec = (GetTickCount64() - tm_start) / 1000;
	u64 days = sec / (3600 * 24);
	int hours = (int)(sec - days * (3600 * 24)) / 3600;
	int min = (int)(sec - days * (3600 * 24) - hours * 3600) / 60;
	 
	u64 dpcnt = gDpLogFile ? gDpLogCnt : db.GetBlockCnt();
	printf("%sSpeed: %d MKeys/s, Err: %d, DPs: %lluK/%lluK, Time: %llud:%02dh:%02dm/%llud:%02dh:%02dm\r\n", gGenMode ? "GEN: " : (IsBench ? "BENCH: " : "MAIN: "), speed, gTotalErrors, dpcnt/1000, est_dps_cnt/1000, days, hours, min, exp_days, exp_hours, exp_min);
}

static void OnStopSignal(int)
{
	gStopReq = 1;
}

//Header[0] stays Range so tames files keep working. The rest pins a table to one
//target: attaching a table to the wrong key, offset or DP silently poisons it.
static void StampDbHeader()
{
	memset(db.Header, 0, sizeof(db.Header));
	db.Header[0] = (u8)gRange;
	db.Header[1] = (u8)gDP;
	db.Header[2] = DBHDR_MARK;
	db.Header[3] = DBHDR_VER;
	memcpy(db.Header + 8, gPubKey.x.data, 32);
	memcpy(db.Header + 40, gStart.data, 32);

	//past byte 72: live stats, so a reader gets totals from the 256-byte header
	//instead of parsing a table that grows into the tens of GB
	u64 dpcnt = db.GetBlockCnt();
	u64 ops = gPrevOps + PntTotalOps;
	u64 tm = (u64)time(NULL);
	memcpy(db.Header + 72, &dpcnt, 8);
	memcpy(db.Header + 80, &ops, 8);
	memcpy(db.Header + 88, &tm, 8);
}

//Returns false if the file on disk belongs to a different search.
static bool CheckDbHeader()
{
	u8 want[256];
	memcpy(want, db.Header, sizeof(want));
	StampDbHeader();
	bool ok = (memcmp(want, db.Header, 72) == 0);
	memcpy(db.Header, want, sizeof(want));
	return ok;
}

//A DP log is the alternative to keeping a table: the solver appends every DP and
//never holds one, so its memory is flat no matter how long the run goes. Nothing
//in the search reads the table back, only HandleRec does, so dropping it costs
//nothing but the ability to notice a collision the moment it happens. The
//matcher finds it later, and later is free: a collision, once it is in the data,
//is permanent.
//
//Appending to a log built for another key would silently mix two searches, so an
//existing log's header must agree before we add to it.
static bool OpenDpLog()
{
	u8 want[256];
	memset(want, 0, sizeof(want));
	want[0] = (u8)gRange;
	want[1] = (u8)gDP;
	want[2] = DPLOG_MARK;
	want[3] = DBHDR_VER;
	//the table header keeps only x, so a reader has to guess the parity back;
	//a log is written by something that knows it, so it says.
	want[4] = gGenMode ? 0 : (u8)(1 + (gPubKey.y.data[0] & 1));
	if (!gGenMode)
	{
		memcpy(want + 8, gPubKey.x.data, 32);
		memcpy(want + 40, gStart.data, 32);
	}

	bool exists = IsFileExist(gDpLogName);
	gDpLogFile = fopen(gDpLogName, exists ? "r+b" : "w+b");
	if (!gDpLogFile)
	{
		printf("FATAL: cannot open DP log %s\r\n", gDpLogName);
		return false;
	}
	if (!exists)
	{
		if (fwrite(want, 1, 256, gDpLogFile) != 256)
		{
			printf("FATAL: cannot write DP log header\r\n");
			return false;
		}
	}
	else
	{
		u8 have[256];
		if (fread(have, 1, 256, gDpLogFile) != 256)
		{
			printf("FATAL: DP log %s has no readable header\r\n", gDpLogName);
			return false;
		}
		if (memcmp(have, want, 72))
		{
			printf("FATAL: DP log %s was built for a different key/offset/range/DP.\r\n",
				gDpLogName);
			return false;
		}
		//a crash can leave a partial record. Resume on the record boundary so
		//the next append overwrites the stray bytes; the matcher ignores a
		//partial tail as well, so neither side ever reads half a record.
		k4u_fseek(gDpLogFile, 0, SEEK_END);
		i64 sz = (i64)k4u_ftell(gDpLogFile);
		gDpLogCnt = (u64)((sz - 256) / (i64)sizeof(DBRec));
		memcpy(&gDpLogOps, have + 80, 8);
		i64 whole = 256 + (i64)gDpLogCnt * (i64)sizeof(DBRec);
		if (whole != sz)
			printf("DP log: %lld trailing bytes are a partial record, resuming over them\r\n",
				(long long)(sz - whole));
		k4u_fseek(gDpLogFile, whole, SEEK_SET);
	}
	printf("DP log %s: %llu records so far, appending\r\n", gDpLogName,
		(unsigned long long)gDpLogCnt);
	return true;
}

//The count in the header is a convenience for readers; the matcher takes the
//real one from the file size, so a crash between flushes costs nothing.
static void FlushDpLog()
{
	if (!gDpLogFile)
		return;
	fflush(gDpLogFile);
	i64 pos = k4u_ftell(gDpLogFile);
	k4u_fseek(gDpLogFile, 72, SEEK_SET);
	fwrite(&gDpLogCnt, 1, 8, gDpLogFile);
	u64 ops = gPrevOps + PntTotalOps;
	fwrite(&ops, 1, 8, gDpLogFile);
	fflush(gDpLogFile);
	k4u_fseek(gDpLogFile, pos, SEEK_SET);
}

//Write to a temp file and rename, so a crash mid-save cannot corrupt the table.
static bool SaveDbAtomic()
{
	char tmp[1100];
	snprintf(tmp, sizeof(tmp), "%s.tmp", gDbFileName);
	StampDbHeader();
	if (!db.SaveToFile(tmp))
		return false;
#ifdef _WIN32
	//rename() on Windows refuses to replace an existing file, so every save
	//after the first would fail and leave the table frozen at its first copy.
	return MoveFileExA(tmp, gDbFileName, MOVEFILE_REPLACE_EXISTING | MOVEFILE_WRITE_THROUGH) != 0;
#else
	return rename(tmp, gDbFileName) == 0;
#endif
}

bool SolvePoint(EcPoint PntToSolve, int Range, int DP, EcInt* pk_res)
{
	if ((Range < 32) || (Range > 180))
	{
		printf("Unsupported Range value (%d)!\r\n", Range);
		return false;
	}
	if ((DP < 14) || (DP > 32)) 
	{
		printf("Unsupported DP value (%d)!\r\n", DP);
		return false;
	}

	printf("\r\nSolving point: Range %d bits, DP %d, start...\r\n", Range, DP);
	double ops = 1.15 * pow(2.0, Range / 2.0);
	double dp_val = (double)(1ull << DP);
	u64 total_kangs = GpuKangs[0]->CalcKangCnt();
	for (int i = 1; i < GpuCnt; i++)
		total_kangs += GpuKangs[i]->CalcKangCnt();
	double path_single_kang = ops / total_kangs;
	double DPs_per_kang = path_single_kang / dp_val;
	printf("Estimated DPs per kangaroo (ideal): %.2f.%s\r\n", DPs_per_kang, (DPs_per_kang < 5) ? " DP overhead is big, use less DP value if possible!" : "");

	if (DPs_per_kang < 0.001)
		DPs_per_kang = 0.001;
	double K = 1.15 + (0.07 + 0.76 / sqrt(DPs_per_kang)) / (1 + 0.30 * DPs_per_kang); //real K with DP overhead. Empirical formula.
	printf("Estimated K with DP overhead: %.2f (DP overhead is about %d%%)\r\n", K, int(0.5 + 100 * (K / 1.15 - 1.0)) );
	ops = K * pow(2.0, Range / 2.0);

	double ram = (32 + 4 + 4) * ops / dp_val; //+4 for grow allocation and memory fragmentation
	ram += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
	ram /= (1024 * 1024 * 1024); //GB
	printf("SOTA v2 method, estimated ops: 2^%.3f, RAM for DPs: %.3f GB.\r\n", log2(ops), ram);
	gIsOpsLimit = false;
	double MaxTotalOps = 0.0;
	if (gMax > 0)
	{
		MaxTotalOps = gMax * ops;
		double ram_max = (32 + 4 + 4) * MaxTotalOps / dp_val; //+4 for grow allocation and memory fragmentation
		ram_max += sizeof(TListRec) * 256 * 256 * 256; //3byte-prefix table
		ram_max /= (1024 * 1024 * 1024); //GB
		printf("Max allowed number of ops: 2^%.3f, max RAM for DPs: %.3f GB\r\n", log2(MaxTotalOps), ram_max);
	}



	if (!gGenMode && gTamesFileName[0])
	{
		printf("load tames...\r\n");
		if (db.LoadFromFile(gTamesFileName))
		{
			printf("tames loaded\r\n");
			if (db.Header[0] != gRange)
			{
				printf("loaded tames have different range, they cannot be used, clear\r\n");
				db.Clear();
			}
		}
		else
			printf("tames loading failed\r\n");
	}

	if (gDbFileName[0])
	{
		if (IsFileExist(gDbFileName))
		{
			printf("loading DP table %s ...\r\n", gDbFileName);
			if (!db.LoadFromFile(gDbFileName))
			{
				printf("FATAL: DP table failed to load, refusing to start\r\n");
				return false;
			}
			if (!CheckDbHeader())
			{
				printf("FATAL: DP table was built for a different key/offset/range/DP.\r\n");
				printf("Refusing to start so the existing table is not overwritten.\r\n");
				return false;
			}
			memcpy(&gPrevOps, db.Header + 80, 8);
			printf("DP table loaded: %llu DPs, %.3f banked ops (2^%.3f)\r\n",
				(unsigned long long)db.GetBlockCnt(), (double)gPrevOps,
				gPrevOps ? log2((double)gPrevOps) : 0.0);
		}
		else
			printf("DP table %s not found, starting a new one\r\n", gDbFileName);
	}

	if (gDpLogName[0])
	{
		if (!OpenDpLog())
			return false;
		memcpy(&gPrevOps, &gDpLogOps, 8);
	}

	SetRndSeed(0); //use same seed to make tames from file compatible
	PntTotalOps = 0;
	PntIndex = 0;
//prepare jumps
	EcInt minjump, t;
	minjump.Set(1);
	minjump.ShiftLeft(Range / 2 + 3);
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps1[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps1[i].dist.Add(t);
		EcJumps1[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps1[i].p = ec.MultiplyG(EcJumps1[i].dist);
	}

	minjump.Set(1);
	minjump.ShiftLeft(Range - 10); //large jumps for L1S2 loops. Must be almost RANGE_BITS
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps2[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps2[i].dist.Add(t);
		EcJumps2[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps2[i].p = ec.MultiplyG(EcJumps2[i].dist);
	}

	minjump.Set(1);
	minjump.ShiftLeft(Range - 10 - 2); //large jumps for loops >2
	for (int i = 0; i < JMP_CNT; i++)
	{
		EcJumps3[i].dist = minjump;
		t.RndMax(minjump);
		EcJumps3[i].dist.Add(t);
		EcJumps3[i].dist.data[0] &= 0xFFFFFFFFFFFFFFFE; //must be even
		EcJumps3[i].p = ec.MultiplyG(EcJumps3[i].dist);
	}
	SetRndSeed(GetTickCount64());

	Int_HalfRange.Set(1);
	Int_HalfRange.ShiftLeft(Range - 1);
	Pnt_HalfRange = ec.MultiplyG(Int_HalfRange);
	Pnt_NegHalfRange = Pnt_HalfRange;
	Pnt_NegHalfRange.y.NegModP();
	gPntToSolve = PntToSolve;

	if (gImportCnt)
	{
		for (int i = 0; i < gImportCnt; i++)
			ImportTable(gImportFiles[i]);
		gImportCnt = 0;   //once per process, not on every solved point
		if (gDbFileName[0] && !SaveDbAtomic())
			printf("WARNING: save after import failed\r\n");
	}

//prepare GPUs
	for (int i = 0; i < GpuCnt; i++)
		if (!GpuKangs[i]->Prepare(PntToSolve, Range, DP, EcJumps1, EcJumps2, EcJumps3))
		{
			GpuKangs[i]->Failed = true;
			printf("GPU %d Prepare failed\r\n", GpuKangs[i]->CudaIndex);
		}

	u64 tm0 = GetTickCount64();
	printf("GPUs started...\r\n");

#ifdef _WIN32
	HANDLE thr_handles[MAX_GPU_CNT];
#else
	pthread_t thr_handles[MAX_GPU_CNT];
#endif

	u32 ThreadID;
	gSolved = false;
	ThrCnt = GpuCnt;
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		thr_handles[i] = (HANDLE)_beginthreadex(NULL, 0, kang_thr_proc, (void*)GpuKangs[i], 0, &ThreadID);
#else
		pthread_create(&thr_handles[i], NULL, kang_thr_proc, (void*)GpuKangs[i]);
#endif
	}

	u64 tm_stats = GetTickCount64();
	u64 tm_save = GetTickCount64();
	while (!gSolved)
	{
		CheckNewPoints();
		Sleep(10);
		if (GetTickCount64() - tm_stats > 10 * 1000)
		{
			ShowStats(tm0, ops, dp_val);
			tm_stats = GetTickCount64();
		}

		if (gDpLogFile && (GetTickCount64() - tm_save > (u64)gDbSaveSec * 1000))
		{
			FlushDpLog();
			tm_save = GetTickCount64();
		}

		if (gDbFileName[0] && (GetTickCount64() - tm_save > (u64)gDbSaveSec * 1000))
		{
			u64 t0 = GetTickCount64();
			if (SaveDbAtomic())
				printf("DP table saved: %llu DPs, %llu ms\r\n",
					(unsigned long long)db.GetBlockCnt(), (unsigned long long)(GetTickCount64() - t0));
			else
				printf("WARNING: DP table save failed\r\n");
			tm_save = GetTickCount64();
		}

		if (gStopReq)
		{
			gStoppedByUser = true;
			printf("\r\nstop requested\r\n");
			break;
		}

		if ((MaxTotalOps > 0.0) && (PntTotalOps > MaxTotalOps))
		{
			gIsOpsLimit = true;
			printf("Operations limit reached\r\n");
			break;
		}
	}

	printf("Stopping work ...\r\n");
	for (int i = 0; i < GpuCnt; i++)
		GpuKangs[i]->Stop();
	while (ThrCnt)
		Sleep(10);
	for (int i = 0; i < GpuCnt; i++)
	{
#ifdef _WIN32
		CloseHandle(thr_handles[i]);
#else
		pthread_join(thr_handles[i], NULL);
#endif
	}

	if (gDpLogFile)
	{
		FlushDpLog();
		fclose(gDpLogFile);
		gDpLogFile = NULL;
		printf("DP log closed: %llu records\r\n", (unsigned long long)gDpLogCnt);
	}

	if (gDbFileName[0] && !gSolved)
	{
		printf("saving DP table ...\r\n");
		if (SaveDbAtomic())
			printf("DP table saved: %llu DPs\r\n", (unsigned long long)db.GetBlockCnt());
		else
			printf("WARNING: final DP table save failed\r\n");
	}

	if (gStoppedByUser)
	{
		db.Clear();
		return false;
	}

	if (gIsOpsLimit)
	{
		if (gGenMode)
		{
			printf("saving tames...\r\n");
			db.Header[0] = gRange; 
			if (db.SaveToFile(gTamesFileName))
				printf("tames saved\r\n");
			else
				printf("tames saving failed\r\n");
		}
		db.Clear();
		return false;
	}

	K = (double)PntTotalOps / pow(2.0, Range / 2.0);
	printf("Point solved, K: %.3f (with DP and GPU overheads)\r\n\r\n", K);
	db.Clear();
	*pk_res = gPrivKey;
	return true;
}

bool ParseCommandLine(int argc, char* argv[])
{
	int ci = 1;
	while (ci < argc)
	{
		char* argument = argv[ci];
		ci++;
		if (strcmp(argument, "-gpu") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -gpu option\r\n");
				return false;
			}
			char* gpus = argv[ci];
			ci++;
			memset(gGPUs_Mask, 0, sizeof(gGPUs_Mask));
			for (int i = 0; i < (int)strlen(gpus); i++)
			{
				if ((gpus[i] < '0') || (gpus[i] > '9'))
				{
					printf("error: invalid value for -gpu option\r\n");
					return false;
				}
				gGPUs_Mask[gpus[i] - '0'] = 1;
			}
		}
		else
		if (strcmp(argument, "-dp") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 14) || (val > 60))
			{
				printf("error: invalid value for -dp option\r\n");
				return false;
			}
			gDP = val;
		}
		else
		if (strcmp(argument, "-range") == 0)
		{
			int val = atoi(argv[ci]);
			ci++;
			if ((val < 32) || (val > 170))
			{
				printf("error: invalid value for -range option\r\n");
				return false;
			}
			gRange = val;
		}
		else
		if (strcmp(argument, "-start") == 0)
		{	
			if (!gStart.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -start option\r\n");
				return false;
			}
			ci++;
			gStartSet = true;
		}
		else
		if (strcmp(argument, "-pubkey") == 0)
		{
			if (!gPubKey.SetHexStr(argv[ci]))
			{
				printf("error: invalid value for -pubkey option\r\n");
				return false;
			}
			ci++;
		}
		else
		if (strcmp(argument, "-tames") == 0)
		{
			strcpy(gTamesFileName, argv[ci]);
			ci++;
		}
		else
		if (strcmp(argument, "-import") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -import option\r\n");
				return false;
			}
			if (gImportCnt >= MAX_IMPORTS)
			{
				printf("error: too many -import files\r\n");
				return false;
			}
			strncpy(gImportFiles[gImportCnt], argv[ci], sizeof(gImportFiles[0]) - 1);
			gImportCnt++;
			ci++;
		}
		else
		if (strcmp(argument, "-wildonly") == 0)
		{
			//Every kangaroo a wild. The default herd is one tame per two wilds,
			//but tames are the abundant side when rented workers generate
			//nothing else: this table sat at 87.6% tames, and more of them buy
			//very little. A pure wild herd still solves on its own, because
			//wilds are seeded from -P and only x is stored: two wilds landing
			//on mutually negated points record the same DP, which gives
			//2k = d1 + d2 and is the halving in Collision_SOTA. Tame-tame
			//collisions are the only kind that can never close a loop.
			//no ci++: this flag takes no value, and only value-taking options
			//consume the next argv slot. Incrementing here ate the argument
			//that followed -wildonly.
			gWildOnly = true;
		}
		else
		if (strcmp(argument, "-gentames") == 0)
		{
			//pure tame generation: every kangaroo is d*G, no target point is
			//used at all. Tames cannot solve a key on their own, so this is the
			//mode for untrusted rented hardware. No -pubkey is needed or wanted.
			gGenMode = true;
			gGenTames = true;
		}
		else
		if (strcmp(argument, "-db") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -db option\r\n");
				return false;
			}
			strncpy(gDbFileName, argv[ci], sizeof(gDbFileName) - 1);
			ci++;
		}
		else
		if (strcmp(argument, "-dplog") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -dplog option\r\n");
				return false;
			}
			strncpy(gDpLogName, argv[ci], sizeof(gDpLogName) - 1);
			ci++;
		}
		else
		if (strcmp(argument, "-savesec") == 0)
		{
			if (ci >= argc)
			{
				printf("error: missed value after -savesec option\r\n");
				return false;
			}
			int val = atoi(argv[ci]);
			ci++;
			if (val < 10)
			{
				printf("error: invalid value for -savesec option\r\n");
				return false;
			}
			gDbSaveSec = val;
		}
		else
		if (strcmp(argument, "-max") == 0)
		{
			double val = atof(argv[ci]);
			ci++;
			if (val < 0.001)
			{
				printf("error: invalid value for -max option\r\n");
				return false;
			}
			gMax = val;
		}
		else
		{
			printf("error: unknown option %s\r\n", argument);
			return false;
		}
	}
	if (!gPubKey.x.IsZero())
		if (!gStartSet || !gRange || !gDP)
		{
			printf("error: you must also specify -dp, -range and -start options\r\n");
			return false;
		}
	if (gGenTames)
	{
		if (!gRange || !gDP)
		{
			printf("error: -gentames requires -range and -dp\r\n");
			return false;
		}
		if (!gDbFileName[0])
		{
			printf("error: -gentames requires -db to save the tames\r\n");
			return false;
		}
	}
	if (gDbFileName[0])
	{
		if (gPubKey.x.IsZero() && !gGenTames)
		{
			printf("error: -db requires -pubkey (a benchmark has no table worth keeping)\r\n");
			return false;
		}
		if (gTamesFileName[0])
		{
			printf("error: -db and -tames cannot be used together\r\n");
			return false;
		}
	}
	if (gTamesFileName[0] && !IsFileExist(gTamesFileName))
	{
		if (gMax == 0.0)
		{
			printf("error: you must also specify -max option to generate tames\r\n");
			return false;
		}
		gGenMode = true;
	}
	//Checked last, because gGenMode is also switched on implicitly just above by
	//the -tames/-max flow. Checking earlier let "-wildonly -tames new.dat -max 1"
	//through, and gen mode then won every decision: the kernel added no target,
	//every DP was stamped TAME, and -wildonly survived only in the seeding, which
	//drew all distances from the 34x wild range. The result is a tames file whose
	//distances do not match the range in its own header.
	if (gWildOnly)
	{
		// -wildonly seeds every kangaroo from -P, so it needs the target. It is
		// the opposite of -gentames, which deliberately never sees one.
		if (gGenMode || gGenTames)
		{
			printf("error: -wildonly cannot be combined with tames generation; a gen-mode worker has no target to seed wilds from\r\n");
			return false;
		}
		if (gPubKey.x.IsZero())
		{
			printf("error: -wildonly requires -pubkey (wilds are seeded from the target)\r\n");
			return false;
		}
		if (gMax > 0.0)
		{
			//gMax scales the SOTA v2 ops estimate, which is calibrated for the
			//1/3-2/3 herd. Under -wildonly that estimate is uncalibrated, so
			//-max would stop the search at a number that means nothing.
			printf("error: -wildonly and -max cannot be used together (the ops estimate is not calibrated for an all-wild herd)\r\n");
			return false;
		}
	}
	return true;
}

int main(int argc, char* argv[])
{
#ifdef _DEBUG	
	_CrtSetDbgFlag(_CRTDBG_ALLOC_MEM_DF | _CRTDBG_LEAK_CHECK_DF);
#endif

	//line-buffer stdout: logs are redirected to a file and watched live, and
	//block buffering hides hours of progress until the buffer happens to fill
#ifdef _WIN32
	//MSVC has no line buffering (_IOLBF means full buffering there) and rejects
	//a zero size, so go unbuffered instead.
	setvbuf(stdout, NULL, _IONBF, 0);
#else
	setvbuf(stdout, NULL, _IOLBF, 0);
#endif

	printf("********************************************************************************\r\n");
	printf("*                  RCKangaroo v4.0 (c) 2024-2026 RetiredCoder                  *\r\n");
	printf("********************************************************************************\r\n\r\n");

	printf("This software is free and open-source: https://github.com/RetiredC\r\n");
	printf("It demonstrates fast GPU implementation of SOTA v2 Kangaroo method for solving ECDLP\r\n");
	printf("This version is optimized for 4xxx and 5xxx cards, for older cards use previous versions for best performance!\r\n");
#ifdef _WIN32
	printf("Windows version\r\n");
#else
	printf("Linux version\r\n");
#endif

#ifdef DEBUG_MODE
	printf("DEBUG MODE\r\n\r\n");
#endif

	InitEc();
	SetRndSeed(GetTickCount64());
	gDP = 0;
	gRange = 0;
	gStartSet = false;
	gTamesFileName[0] = 0;
	gMax = 0.0;
	gGenMode = false;
	gIsOpsLimit = false;
	memset(gGPUs_Mask, 1, sizeof(gGPUs_Mask));
	if (!ParseCommandLine(argc, argv))
		return 0;

	//-dplog means "keep no table at all", so it cannot share a run with the two
	//things that need one. Silently doing both would cost the memory the log
	//exists to avoid.
	if (gDpLogName[0] && gDbFileName[0])
	{
		printf("error: -dplog and -db are alternatives; a log keeps no table\r\n");
		return 0;
	}
	if (gDpLogName[0] && gImportCnt)
	{
		printf("error: -import merges into a table, so it cannot be used with -dplog\r\n");
		return 0;
	}

	InitGpus();

	if (!GpuCnt)
	{
		printf("No supported GPUs detected, exit\r\n");
		return 0;
	}

	pPntList = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
	pPntList2 = (u8*)malloc(MAX_CNT_LIST * GPU_DP_SIZE);
	TotalOps = 0;
	TotalSolved = 0;
	gTotalErrors = 0;
	IsBench = gPubKey.x.IsZero();

	//graceful stop: bank the DP table instead of losing it to a kill
	signal(SIGINT, OnStopSignal);
	signal(SIGTERM, OnStopSignal);
#ifdef SIGBREAK
	signal(SIGBREAK, OnStopSignal);	//Ctrl+Break on Windows
#endif

	if (!IsBench && !gGenMode)
	{
		printf("\r\nMAIN MODE%s\r\n\r\n", gWildOnly ? " (WILD-ONLY)" : "");
		if (gWildOnly)
			printf("wild-only herd: no local tames. The tame side must come from -db or -tames,\r\n"
			       "and the ops estimate below is the 1/3-2/3 figure, not calibrated for this.\r\n\r\n");
		EcPoint PntToSolve, PntOfs;
		EcInt pk, pk_found;

		PntToSolve = gPubKey;
		if (!gStart.IsZero())
		{
			PntOfs = ec.MultiplyG(gStart);
			PntOfs.y.NegModP();
			PntToSolve = ec.AddPoints(PntToSolve, PntOfs);
		}
		x32.Set(1);
		x32.ShiftLeft(gRange - 5);
		Pntx32 = ec.MultiplyG(x32);
		PntToSolve = ec.AddPoints(PntToSolve, Pntx32); //for smooth edges

		char sx[100], sy[100];
		gPubKey.x.GetHexStr(sx);
		gPubKey.y.GetHexStr(sy);
		printf("Solving public key\r\nX: %s\r\nY: %s\r\n", sx, sy);
		gStart.GetHexStr(sx);
		printf("Offset: %s\r\n", sx);

		if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
		{
			if (!gIsOpsLimit && !gStoppedByUser)
				printf("FATAL ERROR: SolvePoint failed\r\n");
			goto label_end;
		}
		pk_found.AddModP(gStart);
		pk_found.Sub(x32);
		EcPoint tmp = ec.MultiplyG(pk_found);
		if (!tmp.IsEqual(gPubKey))
		{
			printf("FATAL ERROR: SolvePoint found incorrect key\r\n");
			goto label_end;
		}
		//happy end
		char s[100];
		pk_found.GetHexStr(s);
		printf("\r\nPRIVATE KEY: %s\r\n\r\n", s);
		FILE* fp = fopen("RESULTS.TXT", "a");
		if (fp)
		{
			fprintf(fp, "PRIVATE KEY: %s\n", s);
			fclose(fp);
		}
		else //we cannot save the key, show error and wait forever so the key is displayed
		{
			printf("WARNING: Cannot save the key to RESULTS.TXT!\r\n");
			while (1)
				Sleep(100);
		}
	}
	else
	{
		if (gGenMode)
			printf("\r\nTAMES GENERATION MODE\r\n");
		else
			printf("\r\nBENCHMARK MODE\r\n");

		if (gGenTames)
		{
			//one pass, no target: solve nothing, just accumulate tames until stopped
			EcInt dummy, one;
			one.Set(1);
			EcPoint Dummy = ec.MultiplyG(one); //gen-mode kangaroos never use it
			SolvePoint(Dummy, gRange, gDP, &dummy);
			goto label_end;
		}

		//solve points, show K
		while (1)
		{
			EcInt pk, pk_found;
			EcPoint PntToSolve;

			if (!gRange)
				gRange = 78;
			if (!gDP)
				gDP = 16;

			x32.Set(1);
			x32.ShiftLeft(gRange - 5);
			//generate random pk
			pk.RndBits(gRange);
			pk.Add(x32); //for smooth edges
			PntToSolve = ec.MultiplyG(pk);

			if (!SolvePoint(PntToSolve, gRange, gDP, &pk_found))
			{
				if (!gIsOpsLimit && !gStoppedByUser)
					printf("FATAL ERROR: SolvePoint failed\r\n");
				break;
			}
			pk.Sub(x32); //restore
			pk_found.Sub(x32);
			if (!pk_found.IsEqual(pk))
			{
				printf("FATAL ERROR: Found key is wrong!\r\n");
				break;
			}
			TotalOps += PntTotalOps;
			TotalSolved++;
			u64 ops_per_pnt = TotalOps / TotalSolved;
			double K = (double)ops_per_pnt / pow(2.0, gRange / 2.0);
			printf("Points solved: %d, average K: %.3f (with DP and GPU overheads)\r\n", TotalSolved, K);
			//if (TotalSolved >= 100) break; //dbg
		}
	}
label_end:
	for (int i = 0; i < GpuCnt; i++)
		delete GpuKangs[i];
	DeInitEc();
	free(pPntList2);
	free(pPntList);
}

