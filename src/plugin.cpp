// Deadstation's plugin in the Rally Cross recomp module (PSXRecomp's trusted static plugins: runtime/include/
// mod_plugins.h), activated by mods/preloaded/packages/deadstation.rallycross. Its hooks are listed in game.toml's
// mod_function_entry_funcs.
//
// Draped car shadows. The game's own shadow is the car's bounding box (x/z at 3/4) dropped straight down, its 8
// corners set on the terrain (the car renderer 0x80012FF8, from 0x80014040), drawn as up to 6 dark quads at 50%.
// Here every polygon of the car itself is cast along the sun onto the terrain instead: each vertex goes down the
// sun's direction until it meets the ground, the ground's height being the game's own heightfield (sampled by
// the game's 0x8002C194, interpolated as its 0x8002BC14 does). The flat polygons are drawn at 50% like the
// game's, with the PS1 mask bit so each pixel is darkened once. The game's box is culled (its face table zeroed).
//
#include "cpu_state.h"
extern "C" {
#include "gpu_render.h"
extern uint8_t *g_psx_ram;
extern void (*psx_gpu_list_node_hook)(uint32_t addr);	// patch 0024
}
#include "mod_plugins.h"
#include "psx_cycles.h"
#include "../sdk/brick/host_link.h"
#include "../sdk/brick/psx_sdk.h"
#include "../sdk/brick/replay.h"
#include "conditions.h"
#include "world_edge_lit_frag.h"
#include "world_lit_frag.h"
#include "world_lit_vert.h"
#include "world_vert.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>

extern "C" void psx_dispatch_call(CPUState *cpu, uint32_t addr, uint32_t return_addr);
extern "C" uint32_t interrupts_cycles_to_vblank(void);
extern "C" uint16_t sio_get_pad_buttons_slot(int slot);
extern "C" void psx_gpu_draw_offset(int32_t *x, int32_t *y);		// patch 0024
extern "C" void deadstation_scene_perspective(int on);
extern "C" { extern uint32_t deadstation_game_frames; }
extern "C" void (*psx_gpu_triangle_source_hook)(uint32_t addr);	// patch 0028
extern "C" uint32_t psx_cycle_rate;		// patch 0030
extern "C" uint32_t psx_native_gpu_list(uint32_t addr);
extern "C" { extern double deadstation_frame_period_ms; }	// (sdk/brick/host.cpp)

namespace
{
constexpr const char *Id = "deadstation.rallycross";

// the car renderer's shadow block calls the height function for each box corner; its first call (s1 == 1) comes
// after the car's own polygons are in the ordering table, and before the game touches the scratchpad's part
// matrices: the place to cast ours
constexpr uint32_t Height = 0x8002BC14, ShadowCornerReturn = 0x8001414C;
constexpr uint32_t GridSample = 0x8002C194;	// heightfield sample at a 32-unit grid point (x, z) -> v0
constexpr uint32_t BoxFaces = 0x80053CF8;	// the game's shadow box: 6 quads' corner indices (24 bytes)

// the car renderer's frame (sp at the height call): ordering table entry, packet buffer end, camera matrix
constexpr uint32_t FrameOt = 296, FrameBufferEnd = 304, FrameCamera = 408;
// the car object: its parts (count at +0, 44-byte records from +4) and polygons (44-byte records)
constexpr uint32_t CarParts = 16, CarPolygons = 20, CarShade = 0x68;
// the part matrices the renderer left in the scratchpad (camera x car x part): wheel parts 0-11 by corner, the
// body (12 and up)
constexpr uint32_t Scratch = 0x1F800000;

// the race's light (conditions.h): as chosen in the game's menus; the sun's direction in it, the game's y pointing down
conditions::Look look = conditions::lookFor(conditions::Oasis, conditions::Original, conditions::Day, conditions::Clear);
bool nightRace = false;		// (the AI's slower for it: onAiHands)
// the room our packets leave in the game's buffer for what it draws after
constexpr uint32_t BufferReserve = 0x8000;

// Headlights: world lights. They light the track and what stands on it per pixel (shaders/psx_gpu_spot.frag), the other cars by how much of a
// beam reaches each, and each car in a beam casts its shape along the light onto the ground.
constexpr float BeamReach = 2000.f, BeamSide = 1400.f;
struct CarLight
{
	uint32_t object;
	float x, y, z, fx, fy, fz, rx, rz, up[3], side[3];	// its place, forward, its right along the ground; its own up and across
	float x0, x1, z0, z1;						// what its beam can reach
};
CarLight carLights[8];
int carLightCount = 0;
// a frame's cars natively, by their in-between frames' object (replay.h): their polygons and what they cast
std::vector<DeadstationTriangle> carSolid[8], carBlended[8];
float carViewZ[8] = {};
// A car stands on the ground: the game puts it 24 table entries nearer than it is, 24 x 16 of depth, so it's painted over the ground its wheels go a little into (the physics'
// ground is a heightfield, what's drawn is polygons, and they don't quite agree). Tested so here; what's drawn after it sees it about where it is.
constexpr int8_t CarInFront = -2;
constexpr float CarDepthNearer = 32.f;
std::vector<DeadstationGlow> carDarkens[8], carDepths[8];
int currentCarId = 0;
inline float smooth01(float lo, float hi, float v)
{
	const float t = std::clamp((v - lo) / (hi - lo), 0.f, 1.f);
	return t * t * (3.f - 2.f * t);
}
// how strong a car's beam is at a place, as the shader has it
inline float beamAt(const CarLight& c, float x, float y, float z)
{
	const float dx = x - c.x, dy = y - c.y, dz = z - c.z, along = dx * c.fx + dz * c.fz, across = std::fabs(dx * c.rx + dz * c.rz);
	if (along <= 40.f || along >= BeamReach)
		return 0;
	const float left = 1.f - along / BeamReach, width = 70.f + along * 0.42f;
	const float shape = 1.f - smooth01(width * 0.4f, width, across), spill = 0.24f * (1.f - smooth01(0.f, width * 1.5f, across));
	const float high = std::clamp(1.f - std::fabs(dy - c.fy * along) / 450.f, 0.f, 1.f);
	return (shape + spill) * smooth01(40.f, 160.f, along) * left * std::sqrt(left) * high;
}

struct Vec
{
	double x, y, z;
};

struct Affine	// world = m * v + t
{
	double m[3][3];
	double t[3];
	Vec apply(const Vec& v) const
	{
		return { m[0][0] * v.x + m[0][1] * v.y + m[0][2] * v.z + t[0], m[1][0] * v.x + m[1][1] * v.y + m[1][2] * v.z + t[1],
			m[2][0] * v.x + m[2][1] * v.y + m[2][2] * v.z + t[2] };
	}
};

// (DEADSTATION_RALLYCROSS_WITHOUT=rain,beams,...: a whole race with that layer gone, to see what it costs: what only exists moving doesn't show on a held view)
inline bool without(const char *what)
{
	static const char *const list = std::getenv("DEADSTATION_RALLYCROSS_WITHOUT");
	return list != nullptr && std::strstr(list, what) != nullptr;
}
uint32_t rw(uint32_t a) { return psx_mod_read_word(a); }
int16_t rh(uint32_t a) { return (int16_t)psx_mod_read_half(a); }
uint8_t rb(uint32_t a) { return psx_mod_read_byte(a); }

// a GTE matrix in guest memory: R (3x3 s16, 4.12) at +0, T (3 s32) at +20
void readMatrix(uint32_t a, double r[3][3], double t[3])
{
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			r[i][j] = rh(a + 6 * i + 2 * j) / 4096.0;
		t[i] = (int32_t)rw(a + 20 + 4 * i);
	}
}

bool invert(const double a[3][3], double out[3][3])
{
	double det = a[0][0] * (a[1][1] * a[2][2] - a[1][2] * a[2][1]) - a[0][1] * (a[1][0] * a[2][2] - a[1][2] * a[2][0]) +
		a[0][2] * (a[1][0] * a[2][1] - a[1][1] * a[2][0]);
	if (std::fabs(det) < 1e-9)
		return false;
	double k = 1.0 / det;
	out[0][0] = (a[1][1] * a[2][2] - a[1][2] * a[2][1]) * k;
	out[0][1] = (a[0][2] * a[2][1] - a[0][1] * a[2][2]) * k;
	out[0][2] = (a[0][1] * a[1][2] - a[0][2] * a[1][1]) * k;
	out[1][0] = (a[1][2] * a[2][0] - a[1][0] * a[2][2]) * k;
	out[1][1] = (a[0][0] * a[2][2] - a[0][2] * a[2][0]) * k;
	out[1][2] = (a[0][2] * a[1][0] - a[0][0] * a[1][2]) * k;
	out[2][0] = (a[1][0] * a[2][1] - a[1][1] * a[2][0]) * k;
	out[2][1] = (a[0][1] * a[2][0] - a[0][0] * a[2][1]) * k;
	out[2][2] = (a[0][0] * a[1][1] - a[0][1] * a[1][0]) * k;
	return true;
}

// The heightfield, through the game's own sampler. The track's is 64 x 64 tiles of 16 x 16 cells of 32 units:
// 1024 x 1024 samples (x, z from -16384 up to 16383), kept per track (the tiles don't change in a race).
class Ground
{
public:
	explicit Ground(CPUState *cpu) : cpu(cpu)
	{
		if (samples.empty())
			samples.assign(Size * Size, Unknown);
	}

	// the game's 0x8002BC14 (x, z in world units)
	int height(int x, int z)
	{
		int fx = x & 31, fz = z & 31;
		int h00 = sample(x, z), h10 = sample(x + 32, z), h01 = sample(x, z + 32), h11 = sample(x + 32, z + 32);
		int a = (int16_t)(h00 + ((fx * (h10 - h00)) >> 5));
		int b = (int16_t)(h01 + ((fx * (h11 - h01)) >> 5));
		return (int16_t)(a + ((fz * (b - a)) >> 5));
	}

	static void forget() { samples.assign(Size * Size, Unknown); }

private:
	static constexpr int Size = 1024;
	static constexpr int16_t Unknown = INT16_MIN;

	int sample(int x, int z)
	{
		int gx = (x >> 5) + Size / 2, gz = (z >> 5) + Size / 2;
		if (gx < 0 || gx >= Size || gz < 0 || gz >= Size)
			return 0;	// off the map: the game's sampler says 0
		int16_t& h = samples[(size_t)gz * Size + gx];
		if (h == Unknown)
		{
			cpu->gpr[4] = (uint32_t)(int32_t)(x & ~31);
			cpu->gpr[5] = (uint32_t)(int32_t)(z & ~31);
			cpu->gpr[31] = ShadowCornerReturn;
			filling = true;
			psx_dispatch_call(cpu, GridSample, ShadowCornerReturn);
			filling = false;
			h = (int16_t)cpu->gpr[2];
			if (h == Unknown)
				h = Unknown + 1;
		}
		return h;
	}

	CPUState *cpu;
public:
	static inline bool filling = false;		// (the game's own sampler asked, for the table)
	int gridSample(int x, int z) { return sample(x, z); }
private:
	static std::vector<int16_t> samples;
};
std::vector<int16_t> Ground::samples;
// (the mesh's own surfaces: learnSurfaces)
constexpr int SurfaceSize = 1024;
constexpr int16_t SurfaceNone = INT16_MAX;
std::vector<int16_t> meshFloor, meshRoof;
CPUState *surfaceCpu = nullptr;
inline int16_t surfaceAt(const std::vector<int16_t>& map, float x, float z)
{
	const int gx = ((int)x >> 5) + SurfaceSize / 2, gz = ((int)z >> 5) + SurfaceSize / 2;
	if (map.empty() || gx < 0 || gx >= SurfaceSize || gz < 0 || gz >= SurfaceSize)
		return SurfaceNone;
	return map[(size_t)gz * SurfaceSize + gx];
}

// the guest CPU as the hook found it: our calls into the game's sampler must leave no trace (registers, GTE,
// and the cycles they took: the game's frame is not made any longer by them)
struct SavedCpu
{
	explicit SavedCpu(CPUState *cpu) : cpu(cpu), cycles(psx_cycle_count)
	{
		std::memcpy(gpr, cpu->gpr, sizeof(gpr));
		std::memcpy(gteData, cpu->gte_data, sizeof(gteData));
		std::memcpy(gteCtrl, cpu->gte_ctrl, sizeof(gteCtrl));
		pc = cpu->pc;
		hi = cpu->hi;
		lo = cpu->lo;
		muldiv = cpu->muldiv_ts_done;
		gte = cpu->gte_ts_done;
	}
	~SavedCpu()
	{
		std::memcpy(cpu->gpr, gpr, sizeof(gpr));
		std::memcpy(cpu->gte_data, gteData, sizeof(gteData));
		std::memcpy(cpu->gte_ctrl, gteCtrl, sizeof(gteCtrl));
		cpu->pc = pc;
		cpu->hi = hi;
		cpu->lo = lo;
		cpu->muldiv_ts_done = muldiv;
		cpu->gte_ts_done = gte;
		psx_cycle_count = cycles;
	}
	CPUState *cpu;
	uint64_t cycles, muldiv, gte;
	uint32_t gpr[32], gteData[32], gteCtrl[32], pc, hi, lo;
};

// GPU packets into the game's buffer, each linked into one ordering table entry ahead of what is there
// (so the last added draws first)
class Packets
{
public:
	Packets(uint32_t cursorAddress, uint32_t end, uint32_t ot) : cursorAddress(cursorAddress), end(end), ot(ot)
	{
		cursor = rw(cursorAddress);
	}
	~Packets() { psx_mod_write_word(cursorAddress, cursor); }

	bool room(uint32_t bytes) const { return cursor + bytes + BufferReserve < end; }

	void add(const uint32_t *words, uint32_t count)
	{
		uint32_t p = cursor;
		uint32_t head = rw(ot);
		psx_mod_write_word(p, count << 24 | (head & 0xFFFFFF));
		for (uint32_t i = 0; i < count; i++)
			psx_mod_write_word(p + 4 + 4 * i, words[i]);
		psx_mod_write_word(ot, (head & 0xFF000000) | (p & 0xFFFFFF));
		cursor += 4 + 4 * count;
	}

private:
	uint32_t cursorAddress, end, ot, cursor;
};

bool stats = false;
// the time kept: by the clock, against the panel's frame
using Clock = std::chrono::steady_clock;
Clock::time_point keptFrom {}, workEnded {}, passStarted {};
// DEADSTATION_RALLYCROSS_STATS: where a pass's time goes, in ms
struct PassTimes { double brains, physics, track, cars, drawn, undrawn, list; unsigned drawnCount, undrawnCount; } passTimes {};
struct Timed
{
	double& into;
	Clock::time_point from;
	explicit Timed(double& to) : into(to), from(stats ? Clock::now() : Clock::time_point {}) {}
	~Timed() { if (stats) into += std::chrono::duration<double, std::milli>(Clock::now() - from).count(); }
};
unsigned statCars = 0, statPolygons = 0;
bool boxCulled = false;
bool busy = false;

void castShadow(CPUState *cpu)
{
	uint32_t sp = cpu->gpr[29], car = cpu->gpr[20];
	// what shines on it: the sun (a beam's light is drawn with the world, with nothing to stop it)
	if (!look.shadows)
		return;
	const double sunLength = std::sqrt(look.sun[0] * look.sun[0] + look.sun[1] * look.sun[1] + look.sun[2] * look.sun[2]);
	const Vec sun = { look.sun[0] / sunLength, look.sun[1] / sunLength, look.sun[2] / sunLength };
	uint32_t parts = rw(car + CarParts), polygons = rw(car + CarPolygons);
	uint32_t partCount = rw(parts) & 0xFFFF;
	if (partCount == 0 || partCount > 64)
		return;

	// camera (world -> camera space) and its inverse; the part matrices (part -> camera space)
	double camR[3][3], camT[3], camInv[3][3];
	readMatrix(rw(sp + FrameCamera), camR, camT);
	if (!invert(camR, camInv))
		return;
	Affine toWorld[5];	// part -> world, for the 5 scratchpad matrices
	for (int m = 0; m < 5; m++)
	{
		double r[3][3], t[3];
		readMatrix(Scratch + 32 * m, r, t);
		for (int i = 0; i < 3; i++)
		{
			for (int j = 0; j < 3; j++)
				toWorld[m].m[i][j] = camInv[i][0] * r[0][j] + camInv[i][1] * r[1][j] + camInv[i][2] * r[2][j];
			toWorld[m].t[i] = camInv[i][0] * (t[0] - camT[0]) + camInv[i][1] * (t[1] - camT[1]) +
				camInv[i][2] * (t[2] - camT[2]);
		}
	}
	// the GTE's projection: screen offset (16.16) and the projection plane distance
	double ofx = (int32_t)cpu->gte_ctrl[24] / 65536.0, ofy = (int32_t)cpu->gte_ctrl[25] / 65536.0;
	double h = (uint16_t)cpu->gte_ctrl[26];

	SavedCpu saved(cpu);
	Ground ground(cpu);

	const uint32_t shade = std::max((rw(car + CarShade) >> 3) & 0xFFu, (uint32_t)look.shadowGrey);
	const uint32_t color = shade << 16 | shade << 8 | shade;
	std::vector<DeadstationGlow>& darkens = carDarkens[currentCarId];
	unsigned drawn = 0;
	for (uint32_t p = 0; p < partCount; p++)
	{
		uint32_t part = parts + 4 + 44 * p;
		if (rb(part + 25) == 0)
			continue;
		uint32_t lod = rb(part + 24) & 1;
		uint32_t count = psx_mod_read_half(part + 16 + 2 * lod), first = psx_mod_read_half(part + 20 + 2 * lod);
		if (count == 0 || count > 512)
			continue;
		const Affine& m = toWorld[p >= 12 ? 4 : (p & 3) == 0 ? 2 : (p & 3) == 1 ? 0 : (p & 3) == 2 ? 3 : 1];
		for (uint32_t q = 0; q < count; q++)
		{
			uint32_t poly = polygons + 44 * (first + q);
			int n = rh(poly + 18) == 4 ? 4 : 3;
			Vec w[4];
			for (int v = 0; v < n; v++)
			{
				uint32_t a = poly + 12 + 8 * v;
				w[v] = m.apply({ (double)rh(a), (double)rh(a + 2), (double)rh(a + 4) });
			}
			// only the polygons the sun shines on: together they cover the car's silhouette
			Vec e1 = { w[1].x - w[0].x, w[1].y - w[0].y, w[1].z - w[0].z };
			Vec e2 = { w[2].x - w[0].x, w[2].y - w[0].y, w[2].z - w[0].z };
			double facing = (e1.y * e2.z - e1.z * e2.y) * sun.x + (e1.z * e2.x - e1.x * e2.z) * sun.y +
				(e1.x * e2.y - e1.y * e2.x) * sun.z;
			if (facing <= 0)	// (the models wind them so: facing > 0 is lit)
				continue;
			float px[4], py[4], pz[4];
			bool ok = true;
			for (int v = 0; v < n && ok; v++)
			{
				// down the sun's direction to the ground
				const Vec& way = sun;
				double t = 0;
				for (int i = 0; i < 3; i++)
				{
					int y = ground.height((int)std::floor(w[v].x + t * way.x), (int)std::floor(w[v].z + t * way.z));
					t = (y - w[v].y) / way.y;
					if (t < 0)
						t = 0;
				}
				Vec g = { w[v].x + t * way.x, 0, w[v].z + t * way.z };
				g.y = ground.height((int)std::floor(g.x), (int)std::floor(g.z));
				// to the screen, as the GTE projects
				double cx = camR[0][0] * g.x + camR[0][1] * g.y + camR[0][2] * g.z + camT[0];
				double cy = camR[1][0] * g.x + camR[1][1] * g.y + camR[1][2] * g.z + camT[1];
				double cz = camR[2][0] * g.x + camR[2][1] * g.y + camR[2][2] * g.z + camT[2];
				if (cz < 16)
				{
					ok = false;
					break;
				}
				double sx = ofx + cx * h / cz, sy = ofy + cy * h / cz;
				if (sx < -1000 || sx > 1000 || sy < -1000 || sy > 1000)
				{
					ok = false;
					break;
				}
				px[v] = (float)sx, py[v] = (float)sy, pz[v] = std::max((float)cz - 24.f, 16.f);	// (a little above what it lies on)
			}
			if (!ok)
				continue;
			darkens.push_back({ { px[0], px[1], px[2] }, { py[0], py[1], py[2] }, { color, color, color }, { pz[0], pz[1], pz[2] } });
			if (n == 4)
				darkens.push_back({ { px[1], px[2], px[3] }, { py[1], py[2], py[3] }, { color, color, color }, { pz[1], pz[2], pz[3] } });
			drawn++;
		}
	}

	if (stats)
		statPolygons += drawn;
}

// DEADSTATION_RALLYCROSS_STATS: cars drawn (the game's frame rate shows in it) and shadow polygons, per 5 s
void onVBlank();

constexpr uint32_t PadRead = 0x800529C4, PadBuffers = 0x80060D10;

bool secondPadForTests = false;
bool splitNative = false;	// the walk of a split screen's view is ours (onSplitWalk)
int onPadRead(struct CPUState *cpu, uint32_t)
{
	const uint32_t port = cpu->gpr[4] & 1;
	const uint32_t buffer = rw(PadBuffers + 4 * port);
	if ((buffer & 0xFF000000u) != 0x80000000u)
		return 0;
	if (port == 0)
	{
		const uint16_t buttons = sio_get_pad_buttons_slot(0);
		psx_mod_write_byte(buffer, 0x00);
		psx_mod_write_byte(buffer + 1, 0x41);
		psx_mod_write_byte(buffer + 2, (uint8_t)(buttons & 0xFF));
		psx_mod_write_byte(buffer + 3, (uint8_t)(buttons >> 8));
		cpu->gpr[2] = 0;
	}
	else if (secondPadForTests)
	{
		// (DEADSTATION_RALLYCROSS_SECOND_PAD: the game keeps its Two Player row locked without one, and a Brick has one pad: this is the only way there, pressing what the first does)
		const uint16_t buttons = sio_get_pad_buttons_slot(0);
		psx_mod_write_byte(buffer, 0x00);
		psx_mod_write_byte(buffer + 1, 0x41);
		psx_mod_write_byte(buffer + 2, (uint8_t)(buttons & 0xFF));
		psx_mod_write_byte(buffer + 3, (uint8_t)(buttons >> 8));
		cpu->gpr[2] = 0;
	}
	else
	{
		psx_mod_write_byte(buffer, 0xFF);
		cpu->gpr[2] = 0xFFFF;
	}
	return 1;
}

void onHeight(struct CPUState *cpu, uint32_t)
{
	if (psx_mod_read_byte(0x8005649B) > 1 && !splitNative)		// (how many players: a split screen that isn't ours keeps the game's own shadows)
		return;
	if (stats && cpu->gpr[31] == ShadowCornerReturn && (cpu->gpr[17] & 0xFF) == 1)
		statCars++;
	if (busy || cpu->gpr[31] != ShadowCornerReturn || (cpu->gpr[17] & 0xFF) != 1)
		return;
	busy = true;
	if (!boxCulled)
	{
		// the game's shadow box: every face degenerate, so the game draws none of it
		for (uint32_t i = 0; i < 24; i++)
			psx_mod_write_byte(BoxFaces + i, 0);
		boxCulled = true;
	}
	castShadow(cpu);
	busy = false;
}

// ---- The race's conditions: a season, a time of day and the weather (conditions.h) ----
//
// Chosen in the game's own menus. Those are data: a page is 36 bytes (its place, the page Back goes to, the byte that keeps its choice, how
// many rows) and 24 bytes a row (its text, the next page, the test that locks it, what highlighting it does, its hint), and nothing on the
// way from Direction to the cars names a page in code. So three more pages go between them, in RAM nothing uses: between the end of the
// game's data (0x800BB3F0) and where it loads everything (0x800BC500). Original / Daytime / Clear is the game as it shipped. Season mode
// has no pages: the year goes by over its races (conditions::calendar).
//
// What they do, once a race: its palette recoloured as its textures load (the last row of the first file); its sky's packets (32 gradient
// quads, 33 flat ones overhead, 64-82 textured panels round the horizon, 4 copies each) recoloured once its track is in; and a colour over
// the world, on the GPU's way in, for the polygons built in the track's walk: not the sky before them, nor the HUD after.
constexpr uint32_t SelGameType = 0x8005648E, SelTrack = 0x8005648F, SelDifficulty = 0x80056491, SelVersion = 0x80056492, SelDirection = 0x80056493,
		SelSeasonStyle = 0x80056495, SeasonRace = 0x800564B8, SeasonSchedules = 0x800563AC;
constexpr uint32_t DirectionPage = 0x80054714, CarPage = 0x80054768;
constexpr uint32_t PageFont = 0x80053E5C, PageStyle = 0x80053DD0, AlwaysOpen = 0x80019464, KeepPreview = 0x80017438, RowHint = 0x800608DC;
constexpr uint32_t FreeRam = 0x800BB400;
constexpr uint32_t RaceLoad = 0x8002AF20, ImageUpload = 0x8002C8BC, FirstTextures = 0x8002B204;
constexpr uint32_t SkyCounts = 0x80060DFC, SkyGradient = 0x80060E3C, SkyOverhead = 0x80060E78, SkyPanels = 0x80060D8C;
constexpr const char *ChoicesFile = "saves/conditions.cfg";

inline void guestWord(uint32_t a, uint32_t v) { *(uint32_t *)(g_psx_ram + (a & 0x1FFFFC)) = v; }
inline uint32_t guestRead(uint32_t a) { return *(const uint32_t *)(g_psx_ram + (a & 0x1FFFFC)); }
inline uint8_t &guestByte(uint32_t a) { return g_psx_ram[a & 0x1FFFFF]; }

struct Pages
{
	bool built = false;
	uint32_t season = 0, time = 0, wet = 0, snow = 0;
	uint32_t choice = 0;		// four bytes, by page
	uint32_t storm = 0, sandstorm = 0, mediumHigh = 0;
	uint8_t kept[4] = {};
} pages;
bool mediumHigh = false;		// (the camera: below)
constexpr const char *CameraFile = "saves/camera.cfg";

// ---- HUD: each part of it can be turned off, in a page of the pause menu ----
//
// The HUD's records say where they are: a part is what lies in its place on the 320 x 240 screen, whichever of the game's routines made it. In the middle of the screen, what's announced
// and the pause menu lie in none of them, and stay.
enum HudPart { HudTach, HudGear, HudSpeed, HudLap, HudPosition, HudRaceTime, HudLapTimes, HudBestLap, HudMap, HudParts };
struct HudArea { const char *name; int16_t x0, y0, x1, y1; };
constexpr HudArea HudAreas[HudParts] = {
	{ "TACH", 4, 172, 76, 238 },
	{ "GEAR", 76, 190, 97, 209 },
	{ "SPEED", 62, 209, 128, 238 },
	{ "LAP", 6, 14, 96, 33 },
	{ "POSITION", 6, 33, 60, 50 },
	{ "RACE TIME", 6, 50, 92, 68 },
	{ "LAP TIMES", 6, 68, 72, 122 },
	{ "BEST LAP", 214, 200, 306, 238 },
	{ "MAP", 226, 18, 310, 104 },
};
constexpr const char *HudFile = "saves/hud.cfg";
uint32_t hudOff = 0;			// a bit a part
uint32_t hudTexts[HudParts] = {}, hudTextOn = 0, hudTextOff = 0, hudTextDone = 0, hudTextHud = 0, blackFill = 0;
inline bool hudHidden(int x, int y)
{
	if (hudOff == 0)
		return false;
	for (int i = 0; i < HudParts; i++)
		if ((hudOff >> i & 1) != 0 && x >= HudAreas[i].x0 && x < HudAreas[i].x1 && y >= HudAreas[i].y0 && y < HudAreas[i].y1)
			return true;
	return false;
}

uint32_t guestText(uint32_t& at, const char *text)
{
	const uint32_t start = at;
	const size_t n = std::strlen(text) + 1;
	std::memcpy(g_psx_ram + (at & 0x1FFFFF), text, n);
	at += (uint32_t)((n + 3) & ~(size_t)3);
	return start;
}
// a page laid out as the game's are: its top by its rows
uint32_t guestPage(uint32_t& at, uint32_t back, uint32_t choice, std::initializer_list<const char *> rows)
{
	uint32_t texts[8] = {}, n = 0;
	for (const char *row : rows)
		texts[n++] = guestText(at, row);
	const uint32_t page = at, top = n >= 5 ? 0x4A : n == 4 ? 0x54 : n == 3 ? 0x5E : 0x68;
	guestWord(page, 0);
	guestWord(page + 4, 0x14u << 24 | top << 16 | 0x2Eu << 8 | 0xFFu);
	guestWord(page + 8, 0);
	guestWord(page + 12, PageStyle);
	guestWord(page + 16, back);
	guestWord(page + 20, choice);
	guestWord(page + 24, n);
	guestWord(page + 28, PageFont);
	guestWord(page + 32, AlwaysOpen);
	for (uint32_t i = 0; i < n; i++)
	{
		const uint32_t row = page + 36 + i * 24;
		guestWord(row, texts[i]);
		guestWord(row + 4, PageFont);
		guestWord(row + 8, CarPage);
		guestWord(row + 12, AlwaysOpen);
		guestWord(row + 16, KeepPreview);
		guestWord(row + 20, RowHint);
	}
	at = page + 36 + n * 24;
	return page;
}
void setNext(uint32_t page, uint32_t next)
{
	const uint32_t n = guestRead(page + 24);
	for (uint32_t i = 0; i < n; i++)
		guestWord(page + 36 + i * 24 + 8, next);
}
void buildPages()
{
	pages.built = true;
	uint32_t at = FreeRam;
	pages.choice = at;
	at += 8;
	if (FILE *f = std::fopen(ChoicesFile, "rb"))
	{
		if (std::fread(pages.kept, 1, 4, f) != 4)
			std::memset(pages.kept, 0, 4);
		std::fclose(f);
	}
	static constexpr uint8_t Rows[4] = { conditions::SeasonCount, conditions::TimeCount, 4, 3 };
	for (uint32_t i = 0; i < 4; i++)
	{
		if (pages.kept[i] >= Rows[i])
			pages.kept[i] = 0;
		guestByte(pages.choice + i) = pages.kept[i];
	}
	pages.mediumHigh = guestText(at, "MEDIUM HIGH");
	for (int i = 0; i < HudParts; i++)
		hudTexts[i] = guestText(at, HudAreas[i].name);
	hudTextOn = guestText(at, "ON"), hudTextOff = guestText(at, "OFF"), hudTextDone = guestText(at, "DONE"), hudTextHud = guestText(at, "HUD");
	blackFill = at, at += 32;
	if (FILE *f = std::fopen(HudFile, "rb"))
	{
		uint8_t kept[2] = {};
		if (std::fread(kept, 1, 2, f) == 2)
			hudOff = (uint32_t)kept[0] | (uint32_t)kept[1] << 8;
		std::fclose(f);
	}
	if (FILE *f = std::fopen(CameraFile, "rb"))
	{
		mediumHigh = std::fgetc(f) == 1;
		std::fclose(f);
	}
	pages.storm = guestText(at, "Storm");
	pages.sandstorm = guestText(at, "Sandstorm");
	pages.season = guestPage(at, DirectionPage, pages.choice, { "Original", "Spring", "Summer", "Autumn", "Winter" });
	pages.time = guestPage(at, pages.season, pages.choice + 1, { "Daytime", "Dawn", "Dusk", "Night" });
	pages.wet = guestPage(at, pages.time, pages.choice + 2, { "Clear", "Rain", "Storm", "Fog" });
	pages.snow = guestPage(at, pages.time, pages.choice + 3, { "Clear", "Snow", "Fog" });
	std::fprintf(stderr, "deadstation rallycross: the conditions pages, %u bytes at %08X\n", at - FreeRam, FreeRam);
}
// a track's pages as the menus reach them: the Stadium, under its roof, has a time of day alone; snow is for a winter's ground, sandstorms
// for the desert
inline bool winterGround(uint8_t track, uint8_t season)
{
	return season == conditions::Winter || (season == conditions::Original && track == conditions::Alpine);
}
void routeMenus()
{
	const uint8_t track = guestByte(SelTrack), season = guestByte(pages.choice);
	const bool indoors = track == conditions::Stadium;
	const uint32_t weather = winterGround(track, season) ? pages.snow : pages.wet, last = indoors ? pages.time : weather;
	setNext(DirectionPage, indoors ? pages.time : pages.season);
	guestWord(pages.time + 16, indoors ? DirectionPage : pages.season);
	setNext(pages.season, pages.time);
	setNext(pages.time, indoors ? CarPage : weather);
	guestWord(CarPage + 16, last);
	guestWord(pages.wet + 36 + 2 * 24, track == conditions::Oasis ? pages.sandstorm : pages.storm);
}

const conditions::SeasonLook *seasonNow = nullptr;
bool skyDue = false, tintWanted = false;
int tintColour[3] = { 256, 256, 256 }, tintBase[3] = { 256, 256, 256 };

struct PacketSpan
{
	uint32_t from, to;
};
std::vector<PacketSpan> buildingWorld, drawingWorld;	// the polygons built in the track's walk: the world

bool tintOn = false;
inline void setTint(bool world)
{
	if (world == tintOn)
		return;
	tintOn = world;
	if (world)
		deadstation_tint(tintColour[0], tintColour[1], tintColour[2]);
	else
		deadstation_tint(256, 256, 256);
}

void forgetTrackCells();
void costNext();
extern bool costing;
extern int costStep;
// a race is about to load
void onRaceLoad(struct CPUState *, uint32_t)
{
	using namespace conditions;
	if (!pages.built)
		buildPages();
	forgetTrackCells();
	// (DEADSTATION_RALLYCROSS_TRACK: this one, whatever a save has unlocked in the menus, to measure them all)
	if (const char *only = std::getenv("DEADSTATION_RALLYCROSS_TRACK"))
		guestByte(SelTrack) = (uint8_t)std::strtoul(only, nullptr, 10);
	const Track track = (Track)std::min<uint8_t>(guestByte(SelTrack), Stadium);
	Conditions c { Original, Day, Clear };
	if (guestByte(SelGameType) == 1)
	{
		const uint32_t schedule = SeasonSchedules + guestByte(SelSeasonStyle) * 24u + guestByte(SelDifficulty) * 8u;
		c = calendar(track, guestByte(SeasonRace), (int)guestRead(schedule), guestByte(SelVersion), guestByte(SelDirection), guestByte(SelSeasonStyle));
	}
	else
	{
		for (uint32_t i = 0; i < 4; i++)
			if (guestByte(pages.choice + i) != pages.kept[i])
			{
				for (uint32_t j = 0; j < 4; j++)
					pages.kept[j] = guestByte(pages.choice + j);
				if (FILE *f = std::fopen(ChoicesFile, "wb"))
				{
					std::fwrite(pages.kept, 1, 4, f);
					std::fclose(f);
				}
				break;
			}
		c.time = (Time)std::min<uint8_t>(pages.kept[1], Night);
		if (track != Stadium)
		{
			c.season = (Season)std::min<uint8_t>(pages.kept[0], Winter);
			static constexpr Weather Wet[4] = { Clear, Rain, Storm, Fog }, Cold[3] = { Clear, Snow, Fog };
			c.weather = winterGround(track, c.season) ? Cold[std::min<uint8_t>(pages.kept[3], 2)] : Wet[std::min<uint8_t>(pages.kept[2], 3)];
		}
	}
	look = lookFor(track, c.season, c.time, c.weather);
	nightRace = c.time == Night;
	seasonNow = c.season != Original && track != Stadium ? &Seasons[track][c.season - 1] : nullptr;
	const float channels[3] = { look.world.r, look.world.g, look.world.b };
	tintWanted = false;
	for (int i = 0; i < 3; i++)
	{
		tintBase[i] = tintColour[i] = std::clamp((int)std::lround(channels[i] * 256), 0, 511);
		tintWanted = tintWanted || tintColour[i] != 256;
	}
	tintWanted = tintWanted || look.lightning;
	skyDue = true;
	// (the track always loads to the same place: nothing but this says the ground under it has changed)
	Ground::forget();
	std::fprintf(stderr, "deadstation rallycross: track %d, season %d, time %d, weather %d; light %.2f %.2f %.2f\n", track, c.season, c.time, c.weather,
			look.world.r, look.world.g, look.world.b);
}

// its first texture file: the track's palette, its last row
void onImageUpload(struct CPUState *cpu, uint32_t)
{
	if (seasonNow == nullptr || cpu->gpr[31] != FirstTextures)
		return;
	const uint32_t image = cpu->gpr[4];
	const uint32_t w = psx_mod_read_half(image + 4), h = psx_mod_read_half(image + 6);
	if (w < 256 || h == 0)
		return;
	uint16_t *row = (uint16_t *)(g_psx_ram + ((image + 8 + (h - 1) * w * 2) & 0x1FFFFE));
	for (int i = 0; i < 256; i++)
		row[i] = conditions::seasonColour(row[i], *seasonNow);
}

// 0xBBGGRR
inline uint32_t skyWord(const conditions::Rgb& c)
{
	const auto byte = [](float v) { return (uint32_t)std::clamp((int)std::lround(v * 255), 0, 255); };
	return byte(c.r) | byte(c.g) << 8 | byte(c.b) << 16;
}
void patchSky()
{
	skyDue = false;
	const uint32_t counts = guestRead(SkyCounts);
	const uint32_t gradients = guestByte(counts), overhead = guestByte(counts + 1), panels = guestByte(counts + 2);
	if (look.skyChanged)
	{
		// each corner by how bright it was on the way from the top of the sky to the horizon
		const uint32_t list = guestRead(SkyGradient);
		const auto bright = [](uint32_t c) { return (float)(c & 255) * 0.3f + (float)((c >> 8) & 255) * 0.59f + (float)((c >> 16) & 255) * 0.11f; };
		float lo = 1e9f, hi = -1e9f;
		for (uint32_t i = 0; i < gradients; i++)
			for (uint32_t j = 0; j < 4; j++)
			{
				const float b = bright(guestRead(list + i * 176 + 32 + 4 + j * 8));
				lo = std::min(lo, b), hi = std::max(hi, b);
			}
		for (uint32_t i = 0; i < gradients; i++)
			for (uint32_t k = 0; k < 4; k++)
				for (uint32_t j = 0; j < 4; j++)
				{
					const uint32_t at = list + i * 176 + 32 + k * 36 + 4 + j * 8, was = guestRead(at);
					const float t = hi > lo ? (bright(was) - lo) / (hi - lo) : 1.f;
					guestWord(at, (was & 0xFF000000u) | skyWord(conditions::mix(look.zenith, look.horizon, t)));
				}
		const uint32_t top = guestRead(SkyOverhead);
		for (uint32_t i = 0; i < overhead; i++)
			for (uint32_t k = 0; k < 4; k++)
			{
				const uint32_t at = top + i * 128 + 32 + k * 24 + 4;
				guestWord(at, (guestRead(at) & 0xFF000000u) | skyWord(look.zenith));
			}
	}
	if (look.hidePanorama)
		guestByte(counts + 2) = 0;
	else if (look.panorama.r != 1 || look.panorama.g != 1 || look.panorama.b != 1)
	{
		// raw textures made modulated: half of full is as they were
		const conditions::Rgb half = look.panorama * conditions::Rgb { 0.5f, 0.5f, 0.5f };
		const uint32_t list = guestRead(SkyPanels);
		for (uint32_t i = 0; i < panels; i++)
			for (uint32_t k = 0; k < 4; k++)
				guestWord(list + i * 196 + 32 + k * 40 + 4, 0x2Cu << 24 | skyWord(half));
	}
}

// ---- What the weather and the night add to a frame ----
//
// All of it after the world's polygons and before the HUD, depth-tested, so hills and cars hide what's behind them; in a one-player race
// (a split screen's views would each want their own). Falling things have no state: each one's place comes from a number and the clock, in
// a box round the camera. Headlights: the track's polygons within a car's beam (found as the track's walk makes them) drawn again, added,
// the surface's own texels by the beam's shape per pixel (shaders/psx_gpu_spot.frag); a soft cone in the air; a glow at each lamp.
constexpr uint32_t CarObjects = 0x8006597C, CarObjectSize = 112, CarRecordAt = 100, CarCount = 0x80060B5C, Players = 0x8005649B;
constexpr uint32_t CarMatrix = 176;		// in the car's record: rotation, then its place (a GTE matrix)

struct Eye
{
	bool valid = false;
	float r[9], t[3], ofx, ofy, h;
	float x, y, z;		// the camera, in the world
	bool project(float wx, float wy, float wz, float& sx, float& sy, float& depth) const
	{
		depth = r[6] * wx + r[7] * wy + r[8] * wz + t[2];
		if (depth < 16.f)
			return false;
		const float q = h / depth;
		sx = ofx + (r[0] * wx + r[1] * wy + r[2] * wz + t[0]) * q;
		sy = ofy + (r[3] * wx + r[4] * wy + r[5] * wz + t[1]) * q;
		return true;
	}
} eye;
std::vector<DeadstationGlow> glowsNow, veilsNow, fallingNow;
uint32_t weatherClock = 0;	// 60ths of a second
uint32_t blanksSeen = 0;
bool effectsDue = false;
float lightningNow = 0;

bool sixty = true, sixtyRunning = false;	// (60 game frames a second: below)
bool tickPass = true;

inline bool effectsOn()
{
	return guestByte(Players) <= 1 && (look.headlights || look.falling != conditions::Nothing || look.fogFar > 0 || look.lightning);
}
inline uint32_t mixBits(uint32_t v)
{
	v ^= v >> 16, v *= 0x7FEB352Du, v ^= v >> 15, v *= 0x846CA68Bu, v ^= v >> 16;
	return v;
}
inline uint32_t glowWord(const conditions::Rgb& c, float k)
{
	const auto byte = [k](float v) { return (uint32_t)std::clamp((int)(v * k * 255), 0, 255); };
	return byte(c.r) | byte(c.g) << 8 | byte(c.b) << 16;
}
// a soft round spot, brightest at its middle
void softSpot(float x, float y, float z, float half, uint32_t rgb)
{
	static constexpr float Round[9][2] = { { 1, 0 }, { 0.707f, 0.707f }, { 0, 1 }, { -0.707f, 0.707f }, { -1, 0 }, { -0.707f, -0.707f }, { 0, -1 }, { 0.707f, -0.707f }, { 1, 0 } };
	const int step = half >= 4.f ? 1 : 2;
	for (int k = 0; k < 8; k += step)
	{
		const float *a = Round[k], *b = Round[k + step];
		glowsNow.push_back({ { x, x + a[0] * half, x + b[0] * half }, { y, y + a[1] * half, y + b[1] * half }, { rgb, 0, 0 }, { z, z, z } });
	}
}

// the cars' lamps for this frame, as the track's walk begins
void prepareLights()
{
	carLightCount = 0;
	if (!look.headlights)
		return;
	const int cars = std::min<int>(guestByte(CarCount), 8);
	for (int i = 0; i < cars; i++)
	{
		const uint32_t m = guestRead(CarObjects + (uint32_t)i * CarObjectSize + CarRecordAt) + CarMatrix;
		if ((m >> 24) != 0x80)
			continue;
		const auto half = [m](uint32_t at) { return (float)*(const int16_t *)(g_psx_ram + ((m + at) & 0x1FFFFE)) / 4096.f; };
		const auto word = [m](uint32_t at) { return (float)(int32_t)guestRead(m + at); };
		CarLight& c = carLights[carLightCount++];
		c.object = CarObjects + (uint32_t)i * CarObjectSize;
		c.x = word(20), c.y = word(24), c.z = word(28);
		c.fx = half(4), c.fy = half(10), c.fz = half(16);		// (the third column: the model's front is its +z)
		c.up[0] = half(2), c.up[1] = half(8), c.up[2] = half(14);
		c.side[0] = half(0), c.side[1] = half(6), c.side[2] = half(12);		// (the first: across the car, leaning as it leans)
		const float flat = std::max(std::sqrt(c.fx * c.fx + c.fz * c.fz), 0.01f);
		c.rx = c.fz / flat, c.rz = -c.fx / flat;
		const float ex = c.x + c.fx * BeamReach, ez = c.z + c.fz * BeamReach;
		c.x0 = std::min(c.x, ex) - BeamSide, c.x1 = std::max(c.x, ex) + BeamSide;
		c.z0 = std::min(c.z, ez) - BeamSide, c.z1 = std::max(c.z, ez) + BeamSide;
	}
}
// the night sky, behind everything (nothing's depth is yet written, so it needs none but what lets it follow the camera in-between)
void addStars()
{
	if (!look.stars || !eye.valid)
		return;
	constexpr float Far = 60000.f;
	glowsNow.clear();
	for (uint32_t i = 0; i < 120; i++)
	{
		const uint32_t a = mixBits(i * 2654435761u + 17u), b = mixBits(a);
		const float turn = (float)(a & 0xFFFF) * (6.2832f / 65536.f), rise = 0.05f + (float)(b & 0xFFFF) / 65536.f * 1.3f;
		const float wx = std::cos(turn) * std::cos(rise) * Far, wz = std::sin(turn) * std::cos(rise) * Far, wy = -std::sin(rise) * Far;
		// (the sky doesn't move with the camera's place)
		const float vz = eye.r[6] * wx + eye.r[7] * wy + eye.r[8] * wz;
		if (vz < 1000.f)
			continue;
		const float q = eye.h / vz, sx = eye.ofx + (eye.r[0] * wx + eye.r[1] * wy + eye.r[2] * wz) * q, sy = eye.ofy + (eye.r[3] * wx + eye.r[4] * wy + eye.r[5] * wz) * q;
		const float k = 0.3f + 0.7f * (float)((b >> 16) & 255) / 255.f;
		softSpot(sx, sy, Far, 0.9f + k * 1.3f, glowWord({ 0.9f, 0.93f, 1.0f }, std::min(k * 1.5f, 1.f)));
	}
	// the moon
	{
		const float wx = 0.5f * Far, wy = -0.62f * Far, wz = 0.6f * Far;
		const float vz = eye.r[6] * wx + eye.r[7] * wy + eye.r[8] * wz;
		if (vz > 1000.f)
		{
			const float q = eye.h / vz, sx = eye.ofx + (eye.r[0] * wx + eye.r[1] * wy + eye.r[2] * wz) * q, sy = eye.ofy + (eye.r[3] * wx + eye.r[4] * wy + eye.r[5] * wz) * q;
			softSpot(sx, sy, Far, 30.f, glowWord({ 0.32f, 0.36f, 0.48f }, 1.f));
			softSpot(sx, sy, Far, 9.f, glowWord({ 1.0f, 1.0f, 0.96f }, 1.f));
		}
	}
	int32_t ox = 0, oy = 0;
	psx_gpu_draw_offset(&ox, &oy);
	for (DeadstationGlow& g : glowsNow)
		for (int j = 0; j < 3; j++)
			g.x[j] += (float)ox, g.y[j] += (float)oy;
	deadstation_replay_object(0);
	deadstation_glows(glowsNow.data(), glowsNow.size());
}

// The weather, in the world. Rain and snow fall down columns fixed in the world, each from high above to where it lands: the ground (the game's heightfield) or the roof of a
// car under it; there rain splashes, a ring on the ground and a crown over it, and snow lies a moment. So it's nearer than a hill and behind a car, you drive through it,
// and a raindrop's streak is its fall as the moving camera sees it: at speed it comes at you. Wheels throw up spray behind a car on the wet road. Sand's still a wind
// through a box round the camera (it doesn't land).
struct Spray
{
	float x, y, z, vx, vy, vz, age, life;
	bool used;
};
Spray sprays[160] = {};
unsigned sprayNext = 0;
uint32_t sprayDiceState = 0x9E3779B9;
inline float sprayDice()
{
	sprayDiceState ^= sprayDiceState << 13, sprayDiceState ^= sprayDiceState >> 17, sprayDiceState ^= sprayDiceState << 5;
	return (float)(sprayDiceState >> 16) / 65536.f;
}
float eyeWas[3] = {}, eyeMoving[3] = {}, carWas[8][3] = {};
uint32_t fallingBlanksWas = 0;
bool carWasKnown[8] = {};

void addFalling(CPUState *cpu)
{
	const float now = (float)((double)weatherClock / 60.0);
	const float dim = std::clamp((look.world.r + look.world.g + look.world.b) / 2.4f, 0.22f, 1.f);
	const bool rain = look.falling == conditions::Drizzle || look.falling == conditions::Downpour;
	if (look.falling == conditions::Sand)
	{
		const auto around = [](float v, float from, float span) {
			v = std::fmod(v - from, span);
			return from + (v < 0 ? v + span : v);
		};
		const conditions::Rgb colour { 0.3f, 0.22f, 0.12f };
		Ground sandGround(cpu);
		for (int i = 0; i < 420; i++)
		{
			const uint32_t a = mixBits((uint32_t)i * 2654435761u + 3u), b = mixBits(a), c = mixBits(b);
			const float phase = (float)(c & 1023u) * 0.00614f;
			const float x = around((float)(a & 0xFFFF) + 1150 * now + 60 * std::sin(now * 0.9f + phase), eye.x - 1500, 3000);
			const float z = around((float)(b & 0xFFFF) + 500 * now + 60 * std::cos(now * 0.7f + phase), eye.z - 1500, 3000);
			// (its height over the ground where it is, not round the camera's; none under a roof)
			if (surfaceAt(meshRoof, x, z) != SurfaceNone)
				continue;
			const float y = (float)sandGround.height((int)x, (int)z) - 8.f - around((float)(c >> 16) + 90 * now, 0, 620);
			float sx, sy, depth;
			if (eye.project(x, y, z, sx, sy, depth) && sx >= -30 && sx <= eye.ofx * 2 + 30 && sy >= -30 && sy <= eye.ofy * 2 + 30)
				softSpot(sx, sy, depth, std::clamp(12.f * eye.h / depth, 0.7f, 24.f), glowWord(colour, std::clamp(1.25f - depth / 1700.f, 0.25f, 1.f) * dim));
		}
		return;
	}
	// the camera's and the cars' motion, per second
	const float seconds = std::max((float)(weatherClock - fallingBlanksWas) / 60.f, 1.f / 60.f);
	fallingBlanksWas = weatherClock;
	const float moved[3] = { (eye.x - eyeWas[0]) / seconds, (eye.y - eyeWas[1]) / seconds, (eye.z - eyeWas[2]) / seconds };
	const bool cut = moved[0] * moved[0] + moved[1] * moved[1] + moved[2] * moved[2] > 12000.f * 12000.f;
	for (int i = 0; i < 3; i++)
		eyeMoving[i] = cut ? 0 : eyeMoving[i] + (moved[i] - eyeMoving[i]) * 0.5f;
	eyeWas[0] = eye.x, eyeWas[1] = eye.y, eyeWas[2] = eye.z;
	struct Roof { float x, z, rx, rz, fx, fz, y; };
	Roof roofs[8];
	int roofCount = 0;
	const int cars = std::min<int>(guestByte(CarCount), 8);
	for (int i = 0; i < cars; i++)
	{
		const uint32_t record = guestRead(CarObjects + (uint32_t)i * CarObjectSize + CarRecordAt);
		if ((record >> 24) != 0x80)
			continue;
		double m[3][3], t[3];
		readMatrix(record + CarMatrix, m, t);
		const float x = (float)t[0], y = (float)t[1], z = (float)t[2];
		const float rx = (float)m[0][0], rz = (float)m[2][0], fx = (float)m[0][2], fz = (float)m[2][2];
		const float dx = x - eye.x, dz = z - eye.z;
		if (dx * dx + dz * dz < 2600.f * 2600.f)
			roofs[roofCount++] = { x, z, rx, rz, fx, fz, y + (float)m[1][1] * -112.f };
		// spray off the rear wheels on the wet road
		if (rain && carWasKnown[i] && tickPass && dx * dx + dz * dz < 6000.f * 6000.f)
		{
			const float vx = (x - carWas[i][0]) / seconds, vz = (z - carWas[i][2]) / seconds, speed = std::sqrt(vx * vx + vz * vz);
			if (speed > 350.f && speed < 12000.f)
				// (one wheel a step, by turns: this is many soft layers of screen, near the camera the whole of it twice over for each car, and it's gone before any frozen view can show it)
				for (int side = (sprayNext & 1) ? -1 : 1, once = 0; once < 1; once++)
				{
					Spray& s = sprays[sprayNext++ % 160];
					const float across = (sprayDice() - 0.5f) * 240.f;
					s = { x + rx * 66.f * (float)side - fx * 130.f, y - 8.f, z + rz * 66.f * (float)side - fz * 130.f,
							vx * 0.42f + rx * across, -(90.f + sprayDice() * 260.f + speed * 0.1f), vz * 0.42f + rz * across, 0, 0.42f + sprayDice() * 0.36f, true };
				}
		}
		carWas[i][0] = x, carWas[i][1] = y, carWas[i][2] = z;
		carWasKnown[i] = true;
	}
	Ground ground(cpu);
	const float sideK = (eye.ofx > 0 ? eye.ofx : 160.f) * 1.3f / eye.h;
	struct Layer { float cell, reach, height, size, strength; bool lands; };
	static const Layer Rain[2][2] = {
		{ { 120, 1700, 1300, 1.3f, 0.92f, true }, { 300, 4600, 2000, 2.4f, 0.46f, false } },
		{ { 82, 1800, 1500, 1.6f, 1.1f, true }, { 235, 5000, 2200, 2.8f, 0.6f, false } },
	};
	static const Layer Snow[2] = { { 110, 1500, 1000, 6.f, 1.f, true }, { 260, 3600, 1400, 8.f, 0.5f, false } };
	const Layer *layers = rain ? Rain[look.falling == conditions::Downpour ? 1 : 0] : Snow;
	const float fallV = look.falling == conditions::Downpour ? 3100.f : look.falling == conditions::Drizzle ? 2300.f : 210.f;
	const float windX = look.falling == conditions::Downpour ? -420.f : look.falling == conditions::Drizzle ? -90.f : 60.f;
	const float windZ = look.falling == conditions::Downpour ? 180.f : look.falling == conditions::Drizzle ? 60.f : 40.f;
	const conditions::Rgb colour = look.falling == conditions::Downpour ? conditions::Rgb { 0.62f, 0.68f, 0.82f } : look.falling == conditions::Drizzle ? conditions::Rgb { 0.54f, 0.6f, 0.72f } : conditions::Rgb { 0.86f, 0.9f, 1.0f };
	constexpr float Exposure = 1.f / 40.f, SplashTime = 0.15f;
	const float *r = eye.r;
	for (int l = 0; l < 2; l++)
	{
		const Layer& layer = layers[l];
		const int x0 = (int)std::floor((eye.x - layer.reach) / layer.cell), x1 = (int)std::floor((eye.x + layer.reach) / layer.cell);
		const int z0 = (int)std::floor((eye.z - layer.reach) / layer.cell), z1 = (int)std::floor((eye.z + layer.reach) / layer.cell);
		const float fallTime = layer.height / fallV, cycle = fallTime + (rain ? SplashTime + 0.04f : 0.6f);
		for (int iz = z0; iz <= z1; iz++)
			for (int ix = x0; ix <= x1; ix++)
			{
				const uint32_t a = mixBits((uint32_t)ix * 73856093u ^ (uint32_t)iz * 19349663u ^ (uint32_t)l * 83492791u), b = mixBits(a);
				const float landX = ((float)ix + (float)(a & 255) / 256.f) * layer.cell, landZ = ((float)iz + (float)((a >> 8) & 255) / 256.f) * layer.cell;
				// (in front of the camera, and within the frame)
				const float vz = r[6] * landX + r[7] * eye.y + r[8] * landZ + eye.t[2];
				if (vz < -300.f)
					continue;
				const float vx = r[0] * landX + r[1] * eye.y + r[2] * landZ + eye.t[0];
				if (std::fabs(vx) > (vz + 500.f) * sideK)
					continue;
				float landY = (float)ground.height((int)landX, (int)landZ);
				// (on a roof, a tunnel's, not through it)
				const int16_t roof = surfaceAt(meshRoof, landX, landZ);
				if (roof != SurfaceNone && (float)roof < landY)
					landY = (float)roof;
				bool onCar = false;
				for (int c = 0; c < roofCount && l == 0; c++)
				{
					const Roof& roof = roofs[c];
					const float dx = landX - roof.x, dz = landZ - roof.z, along = dx * roof.fx + dz * roof.fz, side = dx * roof.rx + dz * roof.rz;
					if (along > -150.f && along < 130.f && std::fabs(side) < 66.f)
					{
						landY = roof.y + (along > 30.f || along < -100.f ? 34.f : 0.f);
						onCar = true;
					}
				}
				const float phased = now / cycle + (float)(b & 0xFFFF) / 65536.f, t = (phased - std::floor(phased)) * cycle;
				if (t < fallTime)
				{
					const float left = fallTime - t, sway = rain ? 0.f : 26.f * std::sin(now * 1.1f + (float)(a >> 16 & 255) * 0.0245f);
					const float x = landX - windX * left + sway, y = landY - fallV * left, z = landZ - windZ * left;
					float sx, sy, depth;
					if (!eye.project(x, y, z, sx, sy, depth) || depth > layer.reach * 1.1f || depth < 90.f)
						continue;
					float k = std::clamp(1.25f - depth / (layer.reach * 0.9f), 0.2f, 1.f) * dim * layer.strength;
					if (depth < 1700.f)
						for (int c = 0; c < carLightCount; c++)
						{
							const CarLight& car = carLights[c];
							const float dx = x - car.x, dz = z - car.z, along = dx * car.fx + dz * car.fz, across = dx * car.rx + dz * car.rz;
							if (along > 0 && along < 1100 && std::fabs(across) < 60 + along * 0.2f && std::fabs(y - car.y) < 200)
								k = std::min(k + (1.f - along / 1100.f) * look.beam * 1.2f, 1.6f);
						}
					if (rain)
					{
						float tx, ty, tz;
						if (!eye.project(x - (windX - eyeMoving[0]) * Exposure, y - (fallV - eyeMoving[1]) * Exposure, z - (windZ - eyeMoving[2]) * Exposure, tx, ty, tz))
							continue;
						const float dx = sx - tx, dy = sy - ty, len = std::max(std::sqrt(dx * dx + dy * dy), 0.01f);
						// (a streak, however near: at arm's length it would be a bar across the screen)
						const float w = std::clamp(layer.size * eye.h / depth, 0.3f, 1.1f), nx = -dy / len * w, ny = dx / len * w;
						const uint32_t rgb = glowWord(colour, k);
						glowsNow.push_back({ { tx - nx, tx + nx, sx - nx }, { ty - ny, ty + ny, sy - ny }, { 0, 0, rgb }, { tz, tz, depth } });
						glowsNow.push_back({ { tx + nx, sx + nx, sx - nx }, { ty + ny, sy + ny, sy - ny }, { 0, rgb, rgb }, { tz, depth, depth } });
					}
					else
						softSpot(sx, sy, depth, std::clamp(layer.size * eye.h / depth, 0.7f, 24.f), glowWord(colour, k));
					continue;
				}
				// it's landed
				if (!layer.lands)
					continue;
				const float age = (t - fallTime) / (cycle - fallTime);
				if (!rain)
				{
					float sx, sy, depth;
					if (eye.project(landX, landY - 3.f, landZ, sx, sy, depth) && depth < layer.reach)
						softSpot(sx, sy, depth, std::clamp(layer.size * eye.h / depth, 0.7f, 18.f), glowWord(colour, (1.f - age) * dim));
					continue;
				}
				if (age >= 1.f)
					continue;
				const float spread = (onCar ? 5.f : 8.f) + (onCar ? 16.f : 34.f) * age, lift = landY - (onCar ? 2.f : 5.f);
				float cx, cy, cz;
				if (!eye.project(landX, lift, landZ, cx, cy, cz) || cz > 2400.f)
					continue;
				const uint32_t rgb = glowWord(colour, (1.f - age) * dim * std::clamp(1.3f - cz / 2400.f, 0.2f, 1.f) * 0.55f);
				// a ring on what it hit
				static constexpr float RingX[5] = { 1, 0, -1, 0, 1 }, RingZ[5] = { 0, 1, 0, -1, 0 };
				float px[4], py[4], pz[4];
				bool all = true;
				for (int j = 0; j < 4 && all; j++)
					all = eye.project(landX + RingX[j] * spread, lift, landZ + RingZ[j] * spread, px[j], py[j], pz[j]);
				if (all)
					for (int j = 0; j < 4; j++)
					{
						const int n = (j + 1) & 3;
						glowsNow.push_back({ { cx, px[j], px[n] }, { cy, py[j], py[n] }, { rgb, 0, 0 }, { cz, pz[j], pz[n] } });
					}
				// and a crown over it, seen from low down
				float ux, uy, uz;
				if (eye.project(landX, lift - (3.f + 9.f * age), landZ, ux, uy, uz))
				{
					const float w = std::max(0.25f, (0.8f + spread * 0.06f) * eye.h / cz);
					glowsNow.push_back({ { cx - w, cx + w, ux }, { cy, cy, uy }, { rgb, rgb, 0 }, { cz, cz, uz } });
				}
			}
	}
	if (!rain)
		return;
	// the spray
	const float seconds2 = seconds;
	// (a car close by throws it over the camera: a whole screen of soft layers. None right at the eye, and no more than about a quarter of the screen's area in all)
	float sprayArea = 0;
	const float sprayMost = eye.ofx * eye.ofy;
	for (Spray& s : sprays)
	{
		if (!s.used)
			continue;
		s.age += seconds2;
		if (s.age >= s.life)
		{
			s.used = false;
			continue;
		}
		s.x += s.vx * seconds2, s.y += s.vy * seconds2, s.z += s.vz * seconds2;
		s.vy += 1500.f * seconds2;
		const float u = s.age / s.life;
		float sx, sy, depth;
		if (eye.project(s.x, s.y, s.z, sx, sy, depth) && depth < 5000.f && depth > 260.f && sprayArea < sprayMost)
			sprayArea += std::pow(std::clamp((10.f + 24.f * u) * eye.h / depth, 0.8f, 18.f), 2.f) * 3.f,
			softSpot(sx, sy, depth, std::clamp((10.f + 24.f * u) * eye.h / depth, 0.8f, 18.f), glowWord({ 0.5f, 0.55f, 0.62f }, (1.f - u) * (1.f - u) * 0.3f * dim));
	}
}

constexpr float LampX = 50, LampY = -52, LampZ = 134, TailX = 45, TailY = -64, TailZ = -156, TailWide = 26, TailHigh = 12;	// (a tail lamp's lens, on the texture: a band low on the back)
// A lamp's light in the air, as a volume: fixed in the world with the car, so it can only ever go where the car points and lean as the car leans. A flat sheet across the car's own width, spreading
// as the beam does, brightest along its axis and nothing at its edges; seen from the side, where that sheet would be a line, a second one upright through the same axis, by how side on it's seen.
// Each with its true depth at each point, so it stops at what's in its way. Drawn once or twice, so it costs a little of the screen, not many layers of it. In rain, snow or fog there's more
// in the air for it to light.
void addCones()
{
	if (look.haze <= 0)
		return;
	static constexpr float Along[7] = { 20, 90, 220, 420, 720, 1120, 1700 };
	for (int i = 0; i < carLightCount; i++)
	{
		const CarLight& c = carLights[i];
		// (how the eye sees the car: which of the sheets it faces)
		const float ex = c.x - eye.x, ey = c.y - eye.y, ez = c.z - eye.z, el = std::max(std::sqrt(ex * ex + ey * ey + ez * ez), 1.f);
		const float facesFlat = std::fabs(ex * c.up[0] + ey * c.up[1] + ez * c.up[2]) / el, facesUpright = std::fabs(ex * c.side[0] + ey * c.side[1] + ez * c.side[2]) / el;
		for (int side = -1; side <= 1; side += 2)
		{
			const float sideX = c.side[0] * LampX * (float)side, sideY = c.side[1] * LampX * (float)side, sideZ = c.side[2] * LampX * (float)side;
			const float lx = c.x + sideX + c.fx * LampZ + c.up[0] * LampY, ly = c.y + sideY + c.fy * LampZ + c.up[1] * LampY, lz = c.z + sideZ + c.fz * LampZ + c.up[2] * LampY;
			const auto sheet = [&](const float *across, float strength) {
				float sx[7], sy[7], depth[7], lo[7][2], hi[7][2];
				bool known[7];
				for (int j = 0; j < 7; j++)
				{
					const float px = lx + c.fx * Along[j], py = ly + c.fy * Along[j] + Along[j] * 0.035f, pz = lz + c.fz * Along[j], half = 14.f + Along[j] * 0.26f;
					float z0, z1;
					known[j] = eye.project(px, py, pz, sx[j], sy[j], depth[j]) &&
							eye.project(px - across[0] * half, py - across[1] * half, pz - across[2] * half, lo[j][0], lo[j][1], z0) &&
							eye.project(px + across[0] * half, py + across[1] * half, pz + across[2] * half, hi[j][0], hi[j][1], z1);
				}
				const auto light = [&](int n) {
					// (gone by the time it reaches the camera: it's where a car behind the eye's beam would be cut off, a hard edge across it)
					const float fade = (1.f - Along[n] / 1700.f) * smooth01(180.f, 900.f, depth[n]);
					return glowWord({ 1.0f, 0.94f, 0.78f }, look.haze * look.beam * fade * fade * 0.42f * strength);
				};
				for (int j = 0; j < 6; j++)
				{
					if (!known[j] || !known[j + 1])
						continue;
					const uint32_t ca = light(j), cb = light(j + 1);
					for (const auto *e : { lo, hi })
					{
						glowsNow.push_back({ { sx[j], e[j][0], sx[j + 1] }, { sy[j], e[j][1], sy[j + 1] }, { ca, 0, cb }, { depth[j], depth[j], depth[j + 1] } });
						glowsNow.push_back({ { e[j][0], e[j + 1][0], sx[j + 1] }, { e[j][1], e[j + 1][1], sy[j + 1] }, { 0, 0, cb }, { depth[j], depth[j + 1], depth[j + 1] } });
					}
				}
			};
			// (from behind a car, as a chasing camera mostly is, the flat one's all there is, at a slant, so it's kept up: from side on it gives way)
			const float flat = 0.55f + 0.45f * facesFlat;
			sheet(c.side, flat);
			if (facesUpright > 0.2f)
				sheet(c.up, smooth01(0.2f, 0.7f, facesUpright));
		}
	}
}
// A frame's world, in the order light wants it: the track's solid polygons before this; the cars, solid, with their depths; what the sun and
// the beams can't reach (their shadows); the beams, on what they reach and not on what's nearer; then what's seen through: the cars' glass.
void emitCarsAndLight()
{
	int32_t ox = 0, oy = 0;
	psx_gpu_draw_offset(&ox, &oy);
	const float fx = (float)ox, fy = (float)oy;
	const auto placedGlows = [fx, fy](std::vector<DeadstationGlow>& glows) {
		for (DeadstationGlow& g : glows)
			for (int j = 0; j < 3; j++)
				g.x[j] += fx, g.y[j] += fy;
	};
	const auto placedPolygons = [fx, fy](std::vector<DeadstationTriangle>& polygons) {
		for (DeadstationTriangle& t : polygons)
			for (int j = 0; j < 3; j++)
				t.x[j] += fx, t.y[j] += fy;
	};
	setTint(false);	// (their corners have the world's light already)
	if (costStep >= 6)
		for (int id = 0; id < 8; id++)
			carSolid[id].clear(), carDepths[id].clear();
	if (costStep >= 5)
		for (int id = 0; id < 8; id++)
			carBlended[id].clear();
	if (costStep >= 4 || without("shadows"))
		for (int id = 0; id < 8; id++)
			carDarkens[id].clear();
	// (the farthest first: each is tested as nearer than it is, and it's the order that keeps one in front of the other)
	int carOrder[8] = { 0, 1, 2, 3, 4, 5, 6, 7 };
	std::sort(carOrder, carOrder + 8, [](int a, int b) { return carViewZ[a] > carViewZ[b]; });
	for (int id : carOrder)
		if (!carSolid[id].empty())
		{
			// (the game links each at the head of the one table entry: the last made is drawn first)
			std::reverse(carSolid[id].begin(), carSolid[id].end());
			placedPolygons(carSolid[id]);
			deadstation_replay_object(id);
			deadstation_triangles(carSolid[id].data(), carSolid[id].size());
			carSolid[id].clear();
			placedGlows(carDepths[id]);
			deadstation_depth_only(carDepths[id].data(), carDepths[id].size());
			carDepths[id].clear();
		}
	for (int id = 0; id < 8; id++)
	{
		if (carDarkens[id].empty())
			continue;
		deadstation_replay_object(id);
		placedGlows(carDarkens[id]);
		deadstation_shadows(carDarkens[id].data(), carDarkens[id].size(), 1);
		carDarkens[id].clear();
	}
	deadstation_replay_object(0);
	if (eye.valid && carLightCount > 0)
	{
		glowsNow.clear();
		if (costStep < 2 && !without("cones"))
			addCones();
		placedGlows(glowsNow);
		deadstation_glows(glowsNow.data(), glowsNow.size());
	}
	for (int id : carOrder)
		if (!carBlended[id].empty())
		{
			std::reverse(carBlended[id].begin(), carBlended[id].end());
			placedPolygons(carBlended[id]);
			deadstation_replay_object(id);
			deadstation_triangles(carBlended[id].data(), carBlended[id].size());
			carBlended[id].clear();
		}
	deadstation_replay_object(0);
}

void addLamps()
{
	for (int i = 0; i < carLightCount; i++)
	{
		const CarLight& c = carLights[i];
		const float dx = eye.x - c.x, dy = eye.y - c.y, dz = eye.z - c.z, dist = std::max(std::sqrt(dx * dx + dy * dy + dz * dz), 1.f);
		const float facing = (dx * c.fx + dy * c.fy + dz * c.fz) / dist;
		for (int side = -1; side <= 1; side += 2)
		{
			const float sideX = c.side[0] * LampX * (float)side, sideY = c.side[1] * LampX * (float)side, sideZ = c.side[2] * LampX * (float)side;
			const float lx = c.x + sideX + c.fx * LampZ + c.up[0] * LampY, ly = c.y + sideY + c.fy * LampZ + c.up[1] * LampY, lz = c.z + sideZ + c.fz * LampZ + c.up[2] * LampY;
			float sx, sy, depth;
			if (facing > 0.1f && eye.project(lx, ly, lz, sx, sy, depth))
			{
				const float glow = facing * facing * look.beam;
				softSpot(sx, sy, depth - 10, std::clamp(30.f * eye.h / depth, 1.5f, 26.f), glowWord({ 1.0f, 0.95f, 0.8f }, glow));
				// on a wet road, its reflection, drawn out towards whoever's looking
				if (look.falling == conditions::Drizzle || look.falling == conditions::Downpour)
				{
					const float toward = std::max(std::sqrt(dx * dx + dz * dz), 1.f), ax = dx / toward, az = dz / toward;
					const float groundY = c.y - 6.f;
					float nx, ny, nz, mx, my, mz, ex, ey, ez, fx2, fy2, fz2;
					if (eye.project(lx - az * 22.f, groundY, lz + ax * 22.f, nx, ny, nz) && eye.project(lx + az * 22.f, groundY, lz - ax * 22.f, mx, my, mz) &&
							eye.project(lx - az * 34.f + ax * 420.f, groundY, lz + ax * 34.f + az * 420.f, ex, ey, ez) && eye.project(lx + az * 34.f + ax * 420.f, groundY, lz - ax * 34.f + az * 420.f, fx2, fy2, fz2))
					{
						const uint32_t rgb = glowWord({ 0.8f, 0.76f, 0.62f }, glow * 0.45f);
						glowsNow.push_back({ { nx, mx, ex }, { ny, my, ey }, { rgb, rgb, 0 }, { nz, mz, ez } });
						glowsNow.push_back({ { mx, fx2, ex }, { my, fy2, ey }, { rgb, 0, 0 }, { mz, fz2, ez } });
					}
				}
			}
			// A tail lamp is a part of the car: a soft disc on its back, by the car's own axes, so it turns, leans and narrows with the body, and from in front it faces away and isn't there.
			const float tx = c.x + c.side[0] * TailX * (float)side + c.fx * TailZ + c.up[0] * TailY, ty = c.y + c.side[1] * TailX * (float)side + c.fy * TailZ + c.up[1] * TailY,
					tz = c.z + c.side[2] * TailX * (float)side + c.fz * TailZ + c.up[2] * TailY;
			float centreX, centreY, centreDepth;
			if ((eye.x - tx) * c.fx + (eye.y - ty) * c.fy + (eye.z - tz) * c.fz < 0.f && eye.project(tx, ty, tz, centreX, centreY, centreDepth))
			{
				static constexpr float Round[9][2] = { { 1, 0 }, { 0.707f, 0.707f }, { 0, 1 }, { -0.707f, 0.707f }, { -1, 0 }, { -0.707f, -0.707f }, { 0, -1 }, { 0.707f, -0.707f }, { 1, 0 } };
				float rimX[9], rimY[9], rimDepth[9];
				bool all = true;
				for (int k = 0; k < 9 && all; k++)
					all = eye.project(tx + c.side[0] * Round[k][0] * TailWide + c.up[0] * Round[k][1] * TailHigh, ty + c.side[1] * Round[k][0] * TailWide + c.up[1] * Round[k][1] * TailHigh,
							tz + c.side[2] * Round[k][0] * TailWide + c.up[2] * Round[k][1] * TailHigh, rimX[k], rimY[k], rimDepth[k]);
				if (all)
				{
					// (the car's depth is this much nearer than it is: on its surface, by that)
					const float nearer = CarDepthNearer + 4.f;
					const uint32_t rgb = glowWord({ 1.0f, 0.1f, 0.06f }, 0.62f);
					for (int k = 0; k < 8; k++)
						glowsNow.push_back({ { centreX, rimX[k], rimX[k + 1] }, { centreY, rimY[k], rimY[k + 1] }, { rgb, 0, 0 },
								{ std::max(centreDepth - nearer, 16.f), std::max(rimDepth[k] - nearer, 16.f), std::max(rimDepth[k + 1] - nearer, 16.f) } });
				}
			}
		}
	}
}

// all of it: the walk of the table has passed the world
void emitEffects()
{
	effectsDue = false;
	if (!eye.valid)
		return;
	int32_t ox = 0, oy = 0;
	psx_gpu_draw_offset(&ox, &oy);
	const float fx = (float)ox, fy = (float)oy;
	deadstation_replay_object(0);
	glowsNow.clear();
	if (look.falling != conditions::Nothing && costStep < 1 && !without("rain"))
		glowsNow.insert(glowsNow.end(), fallingNow.begin(), fallingNow.end());
	addLamps();
	for (DeadstationGlow& g : glowsNow)
		for (int j = 0; j < 3; j++)
			g.x[j] += fx, g.y[j] += fy;
	if (lightningNow > 0)
	{
		// the whole sky lit for a moment
		const uint32_t rgb = glowWord({ 0.62f, 0.68f, 0.86f }, lightningNow);
		const float w = eye.ofx * 2 + fx, h = eye.ofy * 2 + fy;
		glowsNow.push_back({ { fx, w, fx }, { fy, fy, h }, { rgb, rgb, rgb }, { 0, 0, 0 } });
		glowsNow.push_back({ { w, w, fx }, { fy, h, h }, { rgb, rgb, rgb }, { 0, 0, 0 } });
	}
	deadstation_glows(glowsNow.data(), glowsNow.size());
}

// ---- The track natively: 0x800117E4 as host code ----
//
// Each frame the game walks rings of grid cells round the camera, listed for its heading (16 sectors), and for each cell's
// tile draws its objects (the game's own handlers, by type, 0x80058CA4: cars, props) and its polygons (36-byte records: up to
// four vertices, colours in their spare bytes, a texture record). The polygons here take no GTE and no packets: projected
// in host code with the game's matrices, the same tests (facing, the far limit), kept with the entry in the ordering
// table the game would have linked them at, and sent to the GPU as the walk of that table reaches it, so what has no depth
// (the sky, the HUD) lies as the game ordered it. DEADSTATION_SDK_OFF=track: the game's own.
constexpr uint32_t Billboard = 0x80011F18;	// the handler of most of the track's objects
constexpr uint32_t TyreTrack = 0x800129A0, TrackLamp = 0x800121B4, TrackSoundA = 0x8001279C, TrackSoundB = 0x80012844, AnimationsHeld = 0x80060B8C;
constexpr uint32_t TrackDraw = 0x800117E4, TileIndex = 0x80060D74, HeadingLists = 0x80060D78, ArcCos = 0x800583E0, ObjectHandlers = 0x80058CA4;
struct TrackCorner
{
	float sx, sy, z;
	uint32_t rgb;
	uint16_t uv;
};
struct TrackTriangle
{
	TrackCorner c[3];
	uint16_t page, clut, entry;
	uint8_t kind;	// bit 0: raw texture, bit 1: semi-transparent
	int8_t layer;	// (DeadstationTriangle's)
};
// The ordering table's entries a polygon may take: 1535 is the sky's, walked first: anything put with it would go down before the sky and be painted over
constexpr int32_t TrackEntryMost = 1534;
std::vector<TrackTriangle> trackTriangles;	// opaque: drawn nearest first once, the depth test rejecting what's hidden before it's shaded
std::vector<std::pair<float, uint16_t>> trackByDepth;
std::vector<DeadstationWorldQuad> trackQuads, trackQuadsBlended;		// what's drawn with the camera that isn't the mesh: billboards, lamps
std::vector<float> trackQuadsBlendedZ;
std::vector<DeadstationWorldQuad> trackMarks;		// what lies on the ground (tyre tracks): on it, under what stands on it
std::vector<TrackTriangle> trackBlended;	// semi-transparent: at their entries, farthest first, as the game ordered them
uint16_t trackFarthest = 0;
bool trackOpaqueDrawn = false;
uint32_t trackOt = 0;
size_t trackEmitted = 0;
bool trackSorted = false, trackActive = false, inTrack = false;
bool emittingTrack = false;
bool fullDetail = true;
int currentObject = -1;

inline uint32_t ramWord(uint32_t a) { return *(const uint32_t *)(g_psx_ram + (a & 0x1FFFFC)); }
inline int16_t ramHalf(uint32_t a) { return *(const int16_t *)(g_psx_ram + (a & 0x1FFFFE)); }
inline uint8_t ramByte(uint32_t a) { return g_psx_ram[a & 0x1FFFFF]; }
inline int16_t lo16(uint32_t v) { return (int16_t)(v & 0xFFFF); }
inline int16_t hi16(uint32_t v) { return (int16_t)(v >> 16); }

// The cells of the track, and its polygons as one mesh on the GPU (replay.h), given once, a cell a run of it. A cell's polygons reach past its square; its objects stand on it. What's in
// each is the game's detailed list at every distance, with the full textures (+0: +12's half the size, for far off); a few cells have only the plain one: all they are.
struct TrackCell
{
	uint32_t tile;
	int16_t x, z;
	float cx, cy, cz, ex, ey, ez;
};
std::vector<TrackCell> trackCells;
bool trackCellsKnown = false;
std::vector<uint16_t> trackVisible;
DeadstationWorldCamera trackCamera {};
// Two ground polygons in the same place at the same layer: the game has a fixed answer (in one table entry the later in a list is drawn first, so the earlier's on top); depth alone has
// none, and they'd take turns. The later's put a little farther off, for good.
void settleOverlaps(std::vector<DeadstationWorldTriangle>& mesh, const std::vector<int>& cellOfGrid)
{
	struct Flat { float x, y, z, x0, x1, z0, z1; bool ground; };
	std::vector<Flat> flats(mesh.size());
	const size_t cells = trackCells.size();
	std::vector<std::vector<uint32_t>> inCell(cells);
	for (size_t i = 0; i < mesh.size(); i++)
	{
		const DeadstationWorldTriangle& t = mesh[i];
		const float ax = t.p[0][0], ay = t.p[0][1], az = t.p[0][2], bx = t.p[1][0], by = t.p[1][1], bz = t.p[1][2], cx = t.p[2][0], cy = t.p[2][1], cz = t.p[2][2];
		const float ux = bx - ax, uy = by - ay, uz = bz - az, vx = cx - ax, vy = cy - ay, vz = cz - az;
		const float nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
		const float length = nx * nx + ny * ny + nz * nz;
		flats[i] = { (ax + bx + cx) / 3, (ay + by + cy) / 3, (az + bz + cz) / 3, std::min({ ax, bx, cx }), std::max({ ax, bx, cx }), std::min({ az, bz, cz }), std::max({ az, bz, cz }),
				length > 0 && ny * ny > length * 0.25f };
		if (t.cell < cells)
			inCell[t.cell].push_back((uint32_t)i);
	}
	const auto heightAt = [&mesh](uint32_t i, float x, float z, float& y) {
		const DeadstationWorldTriangle& t = mesh[i];
		const float ax = t.p[0][0], az = t.p[0][2], bx = t.p[1][0], bz = t.p[1][2], cx = t.p[2][0], cz = t.p[2][2];
		const float d = (bz - cz) * (ax - cx) + (cx - bx) * (az - cz);
		if (d == 0)
			return false;
		const float l0 = ((bz - cz) * (x - cx) + (cx - bx) * (z - cz)) / d, l1 = ((cz - az) * (x - cx) + (ax - cx) * (z - cz)) / d, l2 = 1 - l0 - l1;
		if (l0 < 0 || l1 < 0 || l2 < 0)
			return false;
		y = l0 * (float)t.p[0][1] + l1 * (float)t.p[1][1] + l2 * (float)t.p[2][1];
		return true;
	};
	for (int gz = 0; gz < 64; gz++)
		for (int gx = 0; gx < 64; gx++)
		{
			const int here = cellOfGrid[gz * 64 + gx];
			if (here < 0)
				continue;
			for (uint32_t a : inCell[(size_t)here])
			{
				const Flat& fa = flats[a];
				if (!fa.ground)
					continue;
				for (int dz = -1; dz <= 1; dz++)
					for (int dx = -1; dx <= 1; dx++)
					{
						const int nz = gz + dz, nx = gx + dx;
						if (nz < 0 || nz >= 64 || nx < 0 || nx >= 64 || cellOfGrid[nz * 64 + nx] < 0)
							continue;
						for (uint32_t b : inCell[(size_t)cellOfGrid[nz * 64 + nx]])
						{
							if (b >= a || !flats[b].ground || mesh[b].layer != mesh[a].layer || fa.x < flats[b].x0 || fa.x > flats[b].x1 || fa.z < flats[b].z0 || fa.z > flats[b].z1)
								continue;
							float y;
							if (heightAt(b, fa.x, fa.z, y) && std::fabs(y - fa.y) < 10.f)
								mesh[a].nudge = std::max<int16_t>(mesh[a].nudge, (int16_t)std::min<int>(mesh[b].nudge + 8, 24));
						}
					}
			}
		}
}
// The mesh's own surfaces on the heightfield's 32-unit grid. The physics' ground and the drawn polygons don't quite agree, which is why a wheel can sink into what's drawn; and the heightfield knows
// nothing of roofs, which is why rain would fall through a tunnel's. Highest first (up is less y).
void learnSurfaces(const std::vector<DeadstationWorldTriangle>& mesh, CPUState *cpu)
{
	meshFloor.assign((size_t)SurfaceSize * SurfaceSize, SurfaceNone);
	meshRoof.assign((size_t)SurfaceSize * SurfaceSize, SurfaceNone);
	const uint32_t ra = cpu->gpr[31];
	Ground ground(cpu);
	for (const DeadstationWorldTriangle& t : mesh)
	{
		const float ax = (float)t.p[1][0] - t.p[0][0], ay = (float)t.p[1][1] - t.p[0][1], az = (float)t.p[1][2] - t.p[0][2];
		const float bx = (float)t.p[2][0] - t.p[0][0], by = (float)t.p[2][1] - t.p[0][1], bz = (float)t.p[2][2] - t.p[0][2];
		const float nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
		const float length = std::sqrt(nx * nx + ny * ny + nz * nz);
		// (not walls, and not what's on end)
		if (length < 1.f || std::fabs(ny) < 0.35f * length)
			continue;
		const float cx = ((float)t.p[0][0] + t.p[1][0] + t.p[2][0]) / 3, cy = ((float)t.p[0][1] + t.p[1][1] + t.p[2][1]) / 3, cz = ((float)t.p[0][2] + t.p[1][2] + t.p[2][2]) / 3;
		const float below = (float)ground.height((int)cx, (int)cz) - cy;
		const bool roof = below > 120.f, floor = std::fabs(below) <= 120.f;
		if (!roof && !floor)
			continue;
		std::vector<int16_t>& map = roof ? meshRoof : meshFloor;
		const int x0 = (std::min({ t.p[0][0], t.p[1][0], t.p[2][0] }) >> 5) + SurfaceSize / 2, x1 = (std::max({ t.p[0][0], t.p[1][0], t.p[2][0] }) >> 5) + SurfaceSize / 2;
		const int z0 = (std::min({ t.p[0][2], t.p[1][2], t.p[2][2] }) >> 5) + SurfaceSize / 2, z1 = (std::max({ t.p[0][2], t.p[1][2], t.p[2][2] }) >> 5) + SurfaceSize / 2;
		const float d = ax * bz - bx * az;
		if (std::fabs(d) < 1.f)
			continue;
		for (int gz = std::max(z0, 0); gz <= std::min(z1, SurfaceSize - 1); gz++)
			for (int gx = std::max(x0, 0); gx <= std::min(x1, SurfaceSize - 1); gx++)
			{
				const float px = (float)((gx - SurfaceSize / 2) * 32 + 16) - t.p[0][0], pz = (float)((gz - SurfaceSize / 2) * 32 + 16) - t.p[0][2];
				const float u = (px * bz - bx * pz) / d, v = (ax * pz - px * az) / d;
				// (a grid point just off its polygon is still its polygon: the wheels look at the nearest, and a roof mustn't leak at its seams)
				if (u < -0.08f || v < -0.08f || u + v > 1.08f)
					continue;
				const int16_t y = (int16_t)((float)t.p[0][1] + u * ay + v * by);
				int16_t& at = map[(size_t)gz * SurfaceSize + gx];
				if (y < at)
					at = y;
			}
	}
	cpu->gpr[31] = ra;
}

// ---- The Lost Mines' lamps give light ----
//
// The game's lamps there are flat glows with no light of their own. Here each flickers orange, on what's near it: the polygons of the track round it, drawn again in its light, added, each with its
// own texture, so the rock's lit and not painted, and how much at each corner by how far and how it faces. Worked out once for a race. What faces away from a lamp (the hillside over a tunnel) gets
// none of it. A car near one takes on the light as a whole.
struct LampCorners { uint32_t triangle; uint8_t w[3]; };
struct LampLight { uint32_t object; float x, y, z, flame, aim; std::vector<LampCorners> lit; };
std::vector<LampLight> lampLights;
const std::vector<DeadstationWorldTriangle> *lampMesh = nullptr;
constexpr float LampReach = 1500.f, LampFlameR = 230.f, LampFlameG = 104.f, LampFlameB = 26.f;
constexpr int LampsLit = 6;
void learnLampLight(const std::vector<DeadstationWorldTriangle>& mesh, uint32_t index)
{
	lampLights.clear();
	lampMesh = &mesh;
	if (guestByte(SelTrack) != conditions::LostMines)
		return;
	for (int cell = 0; cell < 64 * 64; cell++)
	{
		const int32_t tileIndex = ramHalf(index + (uint32_t)cell * 2);
		if (tileIndex == -1)
			continue;
		const uint32_t tile = index + (uint32_t)tileIndex * 88 + 8364;
		for (uint32_t object = ramWord(tile + 68); object != 0; object = ramWord(object + 12))
			if (ramWord(ObjectHandlers + (uint32_t)ramByte(object) * 4) == TrackLamp)
			{
				lampLights.push_back({ object, (float)ramHalf(object + 4), (float)ramHalf(object + 6), (float)ramHalf(object + 8), 1.f, 1.f, {} });
				std::fprintf(stderr, "deadstation rallycross: a lamp at %d %d %d, kind %u, colour %06X\n", ramHalf(object + 4), ramHalf(object + 6), ramHalf(object + 8), ramByte(object + 1), ramWord(object + 20) & 0xFFFFFF);
			}
	}
	// (which way a polygon's corners go round for its face: what's most of a track is ground, and it faces up, y's negative)
	double upward = 0;
	for (const DeadstationWorldTriangle& t : mesh)
		upward += ((double)t.p[1][2] - t.p[0][2]) * ((double)t.p[2][0] - t.p[0][0]) - ((double)t.p[1][0] - t.p[0][0]) * ((double)t.p[2][2] - t.p[0][2]);
	const float wind = upward < 0 ? 1.f : -1.f;
	size_t lit = 0;
	for (LampLight& lamp : lampLights)
	{
		for (uint32_t i = 0; i < (uint32_t)mesh.size(); i++)
		{
			const DeadstationWorldTriangle& t = mesh[i];
			const float ax = (float)t.p[1][0] - t.p[0][0], ay = (float)t.p[1][1] - t.p[0][1], az = (float)t.p[1][2] - t.p[0][2];
			const float bx = (float)t.p[2][0] - t.p[0][0], by = (float)t.p[2][1] - t.p[0][1], bz = (float)t.p[2][2] - t.p[0][2];
			float nx = (ay * bz - az * by) * wind, ny = (az * bx - ax * bz) * wind, nz = (ax * by - ay * bx) * wind;
			const float nl = std::max(std::sqrt(nx * nx + ny * ny + nz * nz), 1.f);
			nx /= nl, ny /= nl, nz /= nl;
			LampCorners corners { i, { 0, 0, 0 } };
			bool any = false;
			for (int j = 0; j < 3; j++)
			{
				const float dx = lamp.x - t.p[j][0], dy = lamp.y - t.p[j][1], dz = lamp.z - t.p[j][2], far = std::sqrt(dx * dx + dy * dy + dz * dz);
				if (far >= LampReach)
					continue;
				const float faces = t.twoSided ? 1.f : (dx * nx + dy * ny + dz * nz) / std::max(far, 1.f);
				if (faces < -0.05f)
					continue;
				const float left = 1.f - far / LampReach;
				corners.w[j] = (uint8_t)std::clamp(left * left * (0.35f + 0.65f * std::max(faces, 0.f)) * 255.f, 0.f, 255.f);
				any = any || corners.w[j] != 0;
			}
			if (any)
				lamp.lit.push_back(corners);
		}
		lit += lamp.lit.size();
	}
	std::fprintf(stderr, "deadstation rallycross: %zu lamps give light, on %zu polygons\n", lampLights.size(), lit);
}
// how much of their light is at a place
inline float lampLightAt(float x, float y, float z)
{
	float light = 0;
	for (const LampLight& lamp : lampLights)
	{
		const float dx = lamp.x - x, dy = lamp.y - y, dz = lamp.z - z, near = dx * dx + dy * dy + dz * dz;
		if (near < LampReach * LampReach)
		{
			const float left = 1.f - std::sqrt(near) / LampReach;
			light += left * left * lamp.flame;
		}
	}
	return std::min(light, 1.2f);
}
void learnTrackCells(uint32_t index)
{
	trackCells.clear();
	trackCellsKnown = true;
	static std::vector<DeadstationWorldTriangle> mesh;
	mesh.clear();
	std::vector<int> cellOfGrid(64 * 64, -1);
	for (int z = 0; z < 64; z++)
		for (int x = 0; x < 64; x++)
		{
			const int32_t tileIndex = ramHalf(index + (uint32_t)(z * 128 + x * 2));
			if (tileIndex == -1)
				continue;
			const uint32_t tile = index + (uint32_t)tileIndex * 88 + 8364;
			const uint32_t list = ramByte(tile + 76) != 0 ? 0 : 1;
			const uint16_t cell = (uint16_t)trackCells.size();
			cellOfGrid[z * 64 + x] = (int)cell;
			float x0 = (float)((x - 32) * 512) - 256, x1 = x0 + 1024, z0 = (float)((z - 32) * 512) - 256, z1 = z0 + 1024, y0 = 1e9f, y1 = -1e9f;
			uint32_t poly = ramWord(tile + 80 + list * 4);
			for (uint32_t count = ramByte(tile + 76 + list); count != 0; count--, poly += 36)
			{
				const uint32_t code = ramByte(poly + 31);
				const bool quad = code == 0x3C || code == 0x2C, smooth = code == 0x3C || code == 0x34;
				int16_t corner[4][3];
				for (int j = 0; j < (quad ? 4 : 3); j++)
				{
					const uint32_t v = poly + (uint32_t)j * 8;
					corner[j][0] = ramHalf(v), corner[j][1] = ramHalf(v + 2), corner[j][2] = ramHalf(v + 4);
					x0 = std::min(x0, (float)corner[j][0]), x1 = std::max(x1, (float)corner[j][0]);
					y0 = std::min(y0, (float)corner[j][1]), y1 = std::max(y1, (float)corner[j][1]);
					z0 = std::min(z0, (float)corner[j][2]), z1 = std::max(z1, (float)corner[j][2]);
				}
				// (a texture record may lie in the scratchpad: the game put the common ones there as the race loaded)
				const uint32_t tex = ramWord(poly + 32), w0 = rw(tex), w1 = rw(tex + 4), w2 = rw(tex + 8);
				const uint16_t uv[4] = { (uint16_t)w0, (uint16_t)w1, (uint16_t)w2, (uint16_t)(w2 >> 16) };
				const uint8_t c0 = ramByte(poly + 22), c1 = ramByte(poly + 23), c2 = ramByte(poly + 30), c3 = ramByte(poly + 7);
				const auto triangle = [&](int a, int b, int c) {
					DeadstationWorldTriangle t {};
					const int k[3] = { a, b, c };
					for (int j = 0; j < 3; j++)
					{
						const int n = k[j];
						t.p[j][0] = corner[n][0], t.p[j][1] = corner[n][1], t.p[j][2] = corner[n][2];
						t.uv[j][0] = (uint8_t)uv[n], t.uv[j][1] = (uint8_t)(uv[n] >> 8);
						if (smooth)
						{
							const uint8_t grey = n == 0 ? c0 : n == 1 ? c1 : n == 2 ? c2 : c3;
							t.rgb[j][0] = t.rgb[j][1] = t.rgb[j][2] = grey;
						}
						else
							t.rgb[j][0] = c0, t.rgb[j][1] = c1, t.rgb[j][2] = c2;
					}
					t.page = (uint16_t)((w1 >> 16) & 0x1FF), t.clut = (uint16_t)(w0 >> 16);
					t.raw = code & 1, t.twoSided = ramByte(poly + 6) != 0;
					// (its place in the game's order, +14 entries of 16 depth each, as a depth: what the game laid over it stays over it. What's laid on another lies within about 64
					// of it, in the tracks' data; more would let a far polygon show through a cliff in front of it)
					t.layer = (int8_t)std::clamp<int32_t>((int8_t)ramByte(poly + 14) / 2, 0, 3);
					t.cell = cell;
					mesh.push_back(t);
				};
				triangle(0, 1, 2);
				if (quad)
					triangle(1, 3, 2);
			}
			if (y0 > y1)
				y0 = -1500, y1 = 1500;
			// (what stands on it, up to the tallest palm: y points down)
			y0 -= 700, y1 += 100;
			trackCells.push_back({ tile, (int16_t)x, (int16_t)z, (x0 + x1) / 2, (y0 + y1) / 2, (z0 + z1) / 2, (x1 - x0) / 2, (y1 - y0) / 2, (z1 - z0) / 2 });
		}
	settleOverlaps(mesh, cellOfGrid);
	deadstation_world_load(mesh.data(), mesh.size(), (unsigned)trackCells.size());
	learnLampLight(mesh, index);
	if (surfaceCpu != nullptr)
		learnSurfaces(mesh, surfaceCpu);
}
// The sky as a mesh, its colours as the race's conditions left them. What the game links last it draws first: overhead, then the gradient, then the far panels.
bool skyKnown = false;
void learnSky()
{
	skyKnown = true;
	static std::vector<DeadstationWorldTriangle> mesh;
	mesh.clear();
	const uint32_t counts = guestRead(SkyCounts);
	const auto corners = [](uint32_t record, DeadstationWorldTriangle& t, int a, int b, int c) {
		const int k[3] = { a, b, c };
		for (int j = 0; j < 3; j++)
			for (int n = 0; n < 3; n++)
				t.p[j][n] = ramHalf(record + (uint32_t)k[j] * 8 + (uint32_t)n * 2);
	};
	const auto colour = [](uint8_t *rgb, uint32_t word) { rgb[0] = (uint8_t)word, rgb[1] = (uint8_t)(word >> 8), rgb[2] = (uint8_t)(word >> 16); };
	const uint32_t top = guestRead(SkyOverhead), gradient = guestRead(SkyGradient), panels = guestRead(SkyPanels);
	for (uint32_t i = 0; i < guestByte(counts + 1); i++)
	{
		const uint32_t record = top + i * 128, word = ramWord(record + 32 + 4);
		for (int half = 0; half < 2; half++)
		{
			DeadstationWorldTriangle t {};
			corners(record, t, half == 0 ? 0 : 1, half == 0 ? 1 : 3, 2);
			for (int j = 0; j < 3; j++)
				colour(t.rgb[j], word);
			t.page = 0xFFFF;
			mesh.push_back(t);
		}
	}
	for (uint32_t i = 0; i < guestByte(counts); i++)
	{
		const uint32_t record = gradient + i * 176, packet = record + 32;
		const uint32_t word[4] = { ramWord(packet + 4), ramWord(packet + 12), ramWord(packet + 20), ramWord(packet + 28) };
		for (int half = 0; half < 2; half++)
		{
			DeadstationWorldTriangle t {};
			const int a = half == 0 ? 0 : 1, b = half == 0 ? 1 : 3;
			corners(record, t, a, b, 2);
			colour(t.rgb[0], word[a]), colour(t.rgb[1], word[b]), colour(t.rgb[2], word[2]);
			t.page = 0xFFFF;
			mesh.push_back(t);
		}
	}
	for (uint32_t i = 0; i < guestByte(counts + 2); i++)
	{
		const uint32_t record = panels + i * 196, packet = record + 32;
		const uint32_t word = ramWord(packet + 4), w12 = ramWord(packet + 12), w20 = ramWord(packet + 20);
		const uint16_t uv[4] = { (uint16_t)w12, (uint16_t)w20, (uint16_t)ramWord(packet + 28), (uint16_t)ramWord(packet + 36) };
		for (int half = 0; half < 2; half++)
		{
			DeadstationWorldTriangle t {};
			const int k[3] = { half == 0 ? 0 : 1, half == 0 ? 1 : 3, 2 };
			corners(record, t, k[0], k[1], k[2]);
			for (int j = 0; j < 3; j++)
			{
				colour(t.rgb[j], word);
				t.uv[j][0] = (uint8_t)uv[k[j]], t.uv[j][1] = (uint8_t)(uv[k[j]] >> 8);
			}
			t.page = (uint16_t)((w20 >> 16) & 0x1FF), t.clut = (uint16_t)(w12 >> 16), t.raw = (word >> 24) & 1;
			mesh.push_back(t);
		}
	}
	deadstation_world_sky_load(mesh.data(), mesh.size());
}
// the game's own sky isn't built
int onGameSky(CPUState *, uint32_t)
{
	// (only where the track's walk was ours: split screens, with their own walk, 0x80014534, keep the game's)
	return fullDetail && trackActive ? 1 : 0;
}
// The beams of the cars nearest the camera, over the cells in their wedges. Each is that much of the screen shaded again, added: the most expensive thing in the frame; a far car's
// pool of light is a few pixels, its lamps (addLamps) what's seen of it.
constexpr int MostBeams = 2;
// The beams for a frame: the nearest car's and the next, each as what the shaders want and the screen it can reach
// a beam: the world to its axes, in world units, each a linear form: x, y, z, constant (shaders/world.vert)
struct WorldBeam { float along[4], across[4], height[4], strength; };
WorldBeam beamsNow[MostBeams];
int beamsCount = 0;
void prepareWorldBeams()
{
	beamsCount = 0;
	if (!eye.valid || costStep >= 3 || without("beams"))
		return;
	int order[8];
	const int lights = std::min(carLightCount, 8);
	for (int i = 0; i < lights; i++)
		order[i] = i;
	const auto away = [](int l) {
		const float dx = carLights[l].x - eye.x, dz = carLights[l].z - eye.z;
		return dx * dx + dz * dz;
	};
	std::sort(order, order + lights, [&](int a, int b) { return away(a) < away(b); });
	const float r31Now = trackCamera.r[6], r32Now = trackCamera.r[7], r33Now = trackCamera.r[8];
	for (int i = 0; i < lights && beamsCount < MostBeams; i++)
	{
		const CarLight& c = carLights[order[i]];
		const float along = c.fx * c.x + c.fz * c.z, across = c.rx * c.x + c.rz * c.z;
		// (none if its wedge is all behind the camera)
		int behind = 0;
		for (int n = 0; n < 8; n++)
		{
			const float out = (n & 1) ? BeamReach : -40.f, side = ((n & 2) ? 1.f : -1.f) * (70.f + 0.42f * std::max(out, 0.f)) * 1.5f, up = (n & 4) ? 450.f : -450.f;
			const float wx = c.x + c.fx * out + c.rx * side, wz = c.z + c.fz * out + c.rz * side, wy = c.y + c.fy * out + up;
			if (r31Now * wx + r32Now * wy + r33Now * wz + trackCamera.t[2] < 16.f)
				behind++;
		}
		if (behind == 8)
			continue;
		beamsNow[beamsCount++] = {
			{ c.fx, 0, c.fz, -along },
			{ c.rx, 0, c.rz, -across },
			{ -c.fy * c.fx, 1, -c.fy * c.fz, -c.y + c.fy * along },
			look.beam,
		};
	}
}
// The headlights in the world's shaders (shaders/world*): the beams as they read them. Strengths: at the corners, the near beam and the other; at the pixels of what's solid, the near one. Then each
// beam's axes. What's solid has the near beam at its pixels, in the one drawing of it; what's cut out and solid quads at their corners.
void worldBeams()
{
	const WorldBeam none {};
	const WorldBeam& near = beamsCount > 0 ? beamsNow[0] : none, &other = beamsCount > 1 ? beamsNow[1] : none;
	float solid[DeadstationWorldNumbers] = { 0, other.strength, near.strength, 0 }, rest[DeadstationWorldNumbers] = { near.strength, other.strength, 0, 0 };
	std::memcpy(solid + 4, near.along, sizeof(float) * 12), std::memcpy(solid + 16, other.along, sizeof(float) * 12);
	std::memcpy(rest + 4, near.along, sizeof(float) * 12), std::memcpy(rest + 16, other.along, sizeof(float) * 12);
	deadstation_world_numbers(solid, rest, near.strength > 0 ? 1 : 0);
}
void forgetTrackCells() { trackCellsKnown = false; skyKnown = false; }
// a lamp's colour for now, by its kind (+1), as the game's handler does each drawing
uint32_t lampRandom = 0x2545F491;
inline uint32_t lampDice()
{
	lampRandom ^= lampRandom << 13, lampRandom ^= lampRandom >> 17, lampRandom ^= lampRandom << 5;
	return lampRandom >> 8;
}
// the lamps near the eye, by their flames of the moment
void addLampLight()
{
	if (lampLights.empty() || lampMesh == nullptr)
		return;
	const bool moving = tickPass && ramByte(AnimationsHeld) == 0;
	int nearest[LampsLit];
	float nearestAt[LampsLit];
	int count = 0;
	for (int i = 0; i < (int)lampLights.size(); i++)
	{
		LampLight& lamp = lampLights[i];
		if (moving)
		{
			// (a flame: restless, now and then all but out)
			const uint32_t dice = lampDice();
			if ((dice & 3) == 0)
				lamp.aim = (dice & 0x1F00) == 0 ? 0.3f : 0.62f + 0.38f * (float)((dice >> 4) & 255) / 255.f;
			lamp.flame += (lamp.aim - lamp.flame) * 0.4f;
		}
		const float dx = lamp.x - eye.x, dy = lamp.y - eye.y, dz = lamp.z - eye.z, at = dx * dx + dy * dy + dz * dz;
		if (at > 9000.f * 9000.f)
			continue;
		if (count == LampsLit && nearestAt[LampsLit - 1] <= at)
			continue;
		int place = count < LampsLit ? count++ : LampsLit - 1;
		while (place > 0 && nearestAt[place - 1] > at)
		{
			nearest[place] = nearest[place - 1], nearestAt[place] = nearestAt[place - 1];
			place--;
		}
		nearest[place] = i, nearestAt[place] = at;
	}
	const float r31 = trackCamera.r[6], r32 = trackCamera.r[7], r33 = trackCamera.r[8], tz = trackCamera.t[2];
	for (int n = 0; n < count; n++)
	{
		const LampLight& lamp = lampLights[nearest[n]];
		for (const LampCorners& corners : lamp.lit)
		{
			const DeadstationWorldTriangle& t = (*lampMesh)[corners.triangle];
			DeadstationWorldQuad quad {};
			float depth = 0;
			for (int j = 0; j < 4; j++)
			{
				const int k = j == 3 ? 2 : j;
				quad.p[j][0] = t.p[k][0], quad.p[j][1] = t.p[k][1], quad.p[j][2] = t.p[k][2];
				quad.uv[j][0] = t.uv[k][0], quad.uv[j][1] = t.uv[k][1];
				const float w = (float)corners.w[k] / 255.f * lamp.flame;
				quad.rgb[j][0] = (uint8_t)(LampFlameR * w), quad.rgb[j][1] = (uint8_t)(LampFlameG * w), quad.rgb[j][2] = (uint8_t)(LampFlameB * w);
				if (j < 3)
					depth += r31 * t.p[k][0] + r32 * t.p[k][1] + r33 * t.p[k][2] + tz;
			}
			if (depth < -3.f * 512.f)
				continue;
			quad.page = t.page, quad.clut = t.clut, quad.raw = 0, quad.semi = 1 + 1;	// (added: the PlayStation's mode 1)
			quad.layer = (int8_t)(t.layer - 1);
			trackQuadsBlended.push_back(quad);
			trackQuadsBlendedZ.push_back(depth / 3.f);
		}
	}
}
void animateLamp(uint32_t o)
{
	uint8_t& r = guestByte(o + 20), &g = guestByte(o + 21), &b = guestByte(o + 22), &aim = guestByte(o + 36);
	switch (guestByte(o + 1))
	{
		case 0:		// the track's colour of the moment
			r = guestByte(0x80060DC8), g = guestByte(0x80060DC0), b = guestByte(0x80060DB8);
			break;
		case 1:		// a flicker
		{
			const uint32_t by = (lampDice() & 0x2F) + 48;
			int8_t& wait = *(int8_t *)&guestByte(o + 39);
			if ((int)(by & 7) < (int)wait)
			{
				if (r >= 161)
					r = (uint8_t)(r - by), g = (uint8_t)(g - by);
				else
					r = (uint8_t)(r + by), g = (uint8_t)(g + by);
				b = g >= 141 ? (uint8_t)(g + 116) : 0;
				wait = 0;
			}
			else
				wait++;
			break;
		}
		case 2:		// white, now and then
			if (r == 0)
			{
				if ((lampDice() & 0x1F) == 0x1F)
				{
					aim = guestByte(o + 37) = guestByte(o + 38) = 0xF0;
					r = g = b = 60;
				}
			}
			else if (r < aim)
				r = g = b = (uint8_t)(r + 60);
			else if (aim < r)
				r = g = b = (uint8_t)(r - 60);
			else
				aim = 0;
			break;
		case 3:		// amber, on and off
			if (r == 0)
			{
				aim = 0xF0, guestByte(o + 37) = 0xC0, guestByte(o + 38) = 0x40;
				r = 60, g = 48, b = 32;
			}
			else if (r < aim)
				r = (uint8_t)(r + 60), g = (uint8_t)(g + 48), b = (uint8_t)(b + 32);
			else if (aim < r)
				r = (uint8_t)(r - 15), g = (uint8_t)(g - 12), b = (uint8_t)(b - 6);
			else
				aim = guestByte(o + 37) = guestByte(o + 38) = 0;
			break;
	}
}

void pauseHighlight(CPUState *cpu);
int onTrackDraw(CPUState *cpu, uint32_t)
{
	if (inTrack)
		return 0;
	if (cpu->gpr[31] == 0x80024E48)
		pauseHighlight(cpu);
	if (skyDue)
		patchSky();
	Timed timedTrack(passTimes.track);
	const uint32_t ra = cpu->gpr[31], sp0 = cpu->gpr[29];
	const uint32_t ot = cpu->gpr[4], cursor = cpu->gpr[5], matrix = cpu->gpr[6], camera = cpu->gpr[7];
	const uint32_t *g = cpu->gte_ctrl;
	const float k = 1.0f / 4096.0f;
	const float r11 = lo16(g[0]) * k, r12 = hi16(g[0]) * k, r13 = lo16(g[1]) * k, r21 = hi16(g[1]) * k, r22 = lo16(g[2]) * k,
			r23 = hi16(g[2]) * k, r31 = lo16(g[3]) * k, r32 = hi16(g[3]) * k, r33 = lo16(g[4]) * k;
	const float tx = (float)(int32_t)g[5], ty = (float)(int32_t)g[6], tz = (float)(int32_t)g[7];
	const float ofx = (int32_t)g[24] / 65536.0f, ofy = (int32_t)g[25] / 65536.0f, h = (float)(uint16_t)g[26];
	if (h <= 0)
		return 0;
	const bool effects = effectsOn();
	if (effects)
	{
		eye.valid = true;
		const float rotation[9] = { r11, r12, r13, r21, r22, r23, r31, r32, r33 };
		std::memcpy(eye.r, rotation, sizeof(rotation));
		eye.t[0] = tx, eye.t[1] = ty, eye.t[2] = tz, eye.ofx = ofx, eye.ofy = ofy, eye.h = h;
		eye.x = (float)(int32_t)psx_mod_read_word(camera + 64), eye.y = (float)(int32_t)psx_mod_read_word(camera + 68), eye.z = (float)(int32_t)psx_mod_read_word(camera + 72);
		prepareLights();
		if (look.falling != conditions::Nothing)
		{
			SavedCpu keep(cpu);
			glowsNow.clear();
			addFalling(cpu);
			fallingNow.swap(glowsNow);
		}
	}
	else
	{
		eye.valid = false;
		carLightCount = 0;
	}
	effectsDue = effects;
	const uint32_t saved[8] = { g[0], g[1], g[2], g[3], g[4], g[5], g[6], g[7] };
	// a frame of guest stack for the handlers: the packet cursor they advance, their fifth argument
	const uint32_t sp = sp0 - 64, cursorAt = sp + 32;
	cpu->gpr[29] = sp;
	psx_mod_write_word(sp + 16, matrix);
	psx_mod_write_word(cursorAt, cursor);
	const uint32_t bufferEnd = cursor + 0x178CC;
	if (trackSorted || trackActive || fullDetail)
	{
		for (int id = 0; id < 8; id++)
			carSolid[id].clear(), carBlended[id].clear(), carDarkens[id].clear(), carDepths[id].clear();
		trackTriangles.clear();
		trackBlended.clear();
		trackMarks.clear();
		trackQuads.clear();
		trackQuadsBlended.clear();
		trackQuadsBlendedZ.clear();
		trackFarthest = 0;
		trackOpaqueDrawn = false;
		trackEmitted = 0;
		trackSorted = false;
	}
	trackOt = ot;
	trackActive = true;
	// Which cells, and what of each. The game's: rings of cells listed for its heading (16 sectors, so 188 of the 640 change as it crosses from one to the next),
	// out to 15572; the near 70 with their detailed polygons and textures, the next 15 with the plain polygons as well, then only the plain, half the far
	// ones' textures at half the size (14 x 14 for 30 x 30). No draw-in: every cell of the track in view, the detailed polygons and the full textures, at any distance.
	struct CellVisit
	{
		uint32_t tile;
		int16_t x, z;
		uint8_t list;
		bool objects, far;
		uint8_t texOffset;
	};
	static std::vector<CellVisit> visits;
	visits.clear();
	trackVisible.clear();
	const uint32_t index = ramWord(TileIndex);
	if (fullDetail)
	{
		trackFarthest = TrackEntryMost;
		const float rotation[9] = { r11, r12, r13, r21, r22, r23, r31, r32, r33 };
		std::memcpy(trackCamera.r, rotation, sizeof(rotation));
		trackCamera.t[0] = tx, trackCamera.t[1] = ty, trackCamera.t[2] = tz, trackCamera.ofx = ofx, trackCamera.ofy = ofy, trackCamera.h = h;
		if (!trackCellsKnown)
		{
			// (0x80023A68: an offset each from 0x80060D84, to a record whose +20 is the rectangle it loads to, by LoadImage)
			static std::vector<int16_t> animated;
			animated.clear();
			surfaceCpu = cpu;
			const uint32_t animations = ramHalf(index + 8356), records = guestRead(0x80060D84);
			for (uint32_t i = 0; i < animations && i < 64; i++)
			{
				const uint32_t record = records + guestRead(records + i * 4);
				for (uint32_t n = 0; n < 4; n++)
					animated.push_back((int16_t)ramHalf(record + 20 + n * 2));
			}
			deadstation_world_animated(animated.data(), animated.size() / 4);
			learnTrackCells(index);
		}
		if (!skyKnown)
			learnSky();
		// (the view's centre needn't be the screen's: as far as the screen goes on its longer side, and some over)
		const float wideK = std::max(ofx, 320.f - ofx) * 1.25f / h, tallK = std::max(ofy, 240.f - ofy) * 1.25f / h;
		const float a11 = std::fabs(r11), a12 = std::fabs(r12), a13 = std::fabs(r13), a21 = std::fabs(r21), a22 = std::fabs(r22), a23 = std::fabs(r23),
				a31 = std::fabs(r31), a32 = std::fabs(r32), a33 = std::fabs(r33);
		for (size_t cellAt = 0; cellAt < trackCells.size(); cellAt++)
		{
			const TrackCell& c = trackCells[cellAt];
			const float vz = r31 * c.cx + r32 * c.cy + r33 * c.cz + tz, ez = a31 * c.ex + a32 * c.ey + a33 * c.ez;
			if (vz + ez < 1.f)
				continue;
			const float vx = r11 * c.cx + r12 * c.cy + r13 * c.cz + tx, ex = a11 * c.ex + a12 * c.ey + a13 * c.ez, reachX = (vz + ez) * wideK;
			if (vx - ex > reachX || vx + ex < -reachX)
				continue;
			const float vy = r21 * c.cx + r22 * c.cy + r23 * c.cz + ty, ey = a21 * c.ex + a22 * c.ey + a23 * c.ez, reachY = (vz + ez) * tallK;
			if (vy - ey > reachY || vy + ey < -reachY)
				continue;
			visits.push_back({ c.tile, c.x, c.z, 0, true, false, 0 });
			trackByDepth.push_back({ vz, (uint16_t)cellAt });
		}
		// (near to far, for what's cut out)
		std::sort(trackByDepth.begin(), trackByDepth.end(), [](const auto& a, const auto& b) { return a.first < b.first; });
		for (const auto& cell : trackByDepth)
			trackVisible.push_back(cell.second);
		trackByDepth.clear();
	}
	else
	{
		// the camera's heading: of 16 sectors, by the game's own table
		const int32_t hx = -(int32_t)ramHalf(matrix + 12), hz = (int32_t)ramHalf(matrix + 16);
		const float len = std::sqrt((float)(hx * hx + hz * hz));
		int32_t cosine = len > 0 ? (int32_t)((float)hz / len * 4096.0f) >> 4 : 256;
		cosine = std::clamp(cosine, -256, 256);
		int32_t sector = (int8_t)ramByte(ArcCos + (uint32_t)(cosine + 256));
		if (hx < 0)
			sector = 256 - sector;
		sector = ((sector + 8) >> 4) & 15;
		const uint32_t lists = ramWord(HeadingLists) + (uint32_t)sector * 1288;
		const int32_t n0 = ramHalf(lists + 1280), n1 = ramHalf(lists + 1282), n3Base = ramHalf(lists + 1284);
		const int32_t cellX = ((int32_t)ramWord(camera + 64) >> 9) + 32, cellZ = ((int32_t)ramWord(camera + 72) >> 9) + 32;
		int32_t n3 = n3Base;
		if (cellX >= 0 && cellX < 64 && cellZ >= 0 && cellZ < 64)
		{
			const int32_t tile = ramHalf(index + (uint32_t)(cellZ * 128 + cellX * 2));
			if (tile != -1)
				n3 -= ramHalf(index + (uint32_t)tile * 88 + 8364 + 78);
		}
		const int32_t counts[4] = { n0, n1, n1, n3 }, starts[4] = { 0, n0, n0, n0 + n1 };
		for (int ring = 0; ring < 4; ring++)
		{
			uint32_t offset = lists + (uint32_t)starts[ring] * 2;
			for (int32_t c = 0; c < counts[ring]; c++, offset += 2)
			{
				const int32_t b0 = (int8_t)ramByte(offset), b1 = (int8_t)ramByte(offset + 1);
				const int32_t x = (b0 >> 1) + cellX, z = b1 + cellZ;
				if (x < 0 || x >= 64 || z < 0 || z >= 64)
					continue;
				const int32_t tileIndex = ramHalf(index + (uint32_t)(z * 128 + x * 2));
				if (tileIndex != -1)
					visits.push_back({ index + (uint32_t)tileIndex * 88 + 8364, (int16_t)x, (int16_t)z, (uint8_t)(ring >= 2 ? 1 : 0), ring != 2, ring >= 2, (uint8_t)((b0 & 1) != 0 && ring != 2 ? 12 : 0) });
			}
		}
	}
	uint32_t drawn = 0;
	inTrack = true;
	const float cameraX = (float)(int32_t)ramWord(camera + 64), cameraY = (float)(int32_t)ramWord(camera + 68), cameraZ = (float)(int32_t)ramWord(camera + 72);
	for (const CellVisit& visit : visits)
	{
		{
			const uint32_t tile = visit.tile, texOffset = visit.texOffset;
			const int32_t x = visit.x, z = visit.z;
			// the beams that can reach it (a tile is 512 a side)
			uint32_t beams = 0;
			for (int l = 0; l < carLightCount; l++)
			{
				const CarLight& car = carLights[l];
				const float x0 = (float)((x - 32) * 512), z0 = (float)((z - 32) * 512);
				if (x0 + 512 >= car.x0 && x0 <= car.x1 && z0 + 512 >= car.z0 && z0 <= car.z1)
					beams |= 1u << l;
			}
			// its objects: the game's handlers (cars, props), with the game's arguments
			if (visit.objects)
				for (uint32_t object = ramWord(tile + 68); object != 0; object = ramWord(object + 12))
				{
					const uint32_t handler = ramWord(ObjectHandlers + (uint32_t)ramByte(object) * 4);
					// a billboard (spectators, palms, banners, grit and smoke), in the world, drawn with the camera; no limit. The game's (0x80011F18) drops any past z 6144 or so, which those
					// at the limit cross and recross as the camera bobs. The way the game has them, turned to face the camera about the vertical only.
					if (handler == Billboard && fullDetail)
					{
						const float mx = (float)ramHalf(matrix), mz = (float)ramHalf(matrix + 4), side = std::sqrt(mx * mx + mz * mz);
						if (side <= 0)
							continue;
						const float wide = (float)(ramHalf(object + 16) >> 5), tall = (float)(ramHalf(object + 18) >> 4);
						const float dx = mx / side * wide, dz = mz / side * wide;
						const float cx = ramHalf(object + 4), cy = ramHalf(object + 6), cz = ramHalf(object + 8);
						const uint32_t colour = ramWord(object + 20), w24 = ramWord(object + 24), w28 = ramWord(object + 28);
						const uint16_t uv32 = (uint16_t)ramHalf(object + 32), uv34 = (uint16_t)ramHalf(object + 34);
						const uint8_t kind2 = (uint8_t)((colour >> 24) & 3);
						const uint16_t page2 = (uint16_t)((w28 >> 16) & 0x1FF);
						DeadstationWorldQuad quad {};
						const float corners[4][3] = { { cx - dx, cy - tall, cz - dz }, { cx + dx, cy - tall, cz + dz }, { cx - dx, cy, cz - dz }, { cx + dx, cy, cz + dz } };
						const uint16_t uvs[4] = { (uint16_t)w24, (uint16_t)w28, uv32, uv34 };
						for (int j = 0; j < 4; j++)
						{
							quad.p[j][0] = (int16_t)corners[j][0], quad.p[j][1] = (int16_t)corners[j][1], quad.p[j][2] = (int16_t)corners[j][2];
							quad.uv[j][0] = (uint8_t)uvs[j], quad.uv[j][1] = (uint8_t)(uvs[j] >> 8);
							quad.rgb[j][0] = (uint8_t)colour, quad.rgb[j][1] = (uint8_t)(colour >> 8), quad.rgb[j][2] = (uint8_t)(colour >> 16);
						}
						quad.page = page2, quad.clut = (uint16_t)(w24 >> 16), quad.raw = kind2 & 1;
						quad.semi = (kind2 & 2) != 0 ? (uint8_t)(1 + ((page2 >> 5) & 3)) : 0;
						(quad.semi != 0 ? trackQuadsBlended : trackQuads).push_back(quad);
						if (quad.semi != 0)
							trackQuadsBlendedZ.push_back(r31 * cx + r32 * cy + r33 * cz + tz);
						continue;
					}
					// a lamp (strings of them, the flat glows that always face the camera), natively: the game's (0x800121B4) drops any past z 6144 or so, crossed
					// and recrossed as the camera bobs; and makes its random flicker at each drawing: here at the game's own 30 a second
					if (handler == TrackLamp && fullDetail)
					{
						if (tickPass && ramByte(AnimationsHeld) == 0)
							animateLamp(object);
						uint32_t colour = ramWord(object + 20);
						if ((colour & 0xFFFFFF) == 0)
							continue;
						const float px = ramHalf(object + 4), py = ramHalf(object + 6), pz = ramHalf(object + 8);
						const float vz = r31 * px + r32 * py + r33 * pz + tz;
						if (vz < 16.f)
							continue;
						// What the game only ever drew near (its limit, z about 6144), left to flash at any distance, is what looks wrong: white flashes all round the track, flat squares
						// flickering far off. Near, as the game has it. Far, a lamp that flickers or pulses is a steady light, and a flash is gone, faded out rather than cut.
						const uint8_t lampKind = ramByte(object + 1);
						if (lampKind != 0)
						{
							const float far = smooth01(4200.f, 6100.f, vz);
							if (lampKind == 2 && far >= 1.f)
								continue;
							const uint32_t steady = lampKind == 1 ? 0xA0u | 0x80u << 8 : lampKind == 3 ? 0x78u | 0x60u << 8 | 0x20u << 16 : 0u;
							const auto mixed = [far](uint32_t a, uint32_t b) { return (uint32_t)((float)a + ((float)b - (float)a) * far); };
							colour = mixed(colour & 255, steady & 255) | mixed((colour >> 8) & 255, (steady >> 8) & 255) << 8 | mixed((colour >> 16) & 255, (steady >> 16) & 255) << 16;
						}
						for (const LampLight& lamp : lampLights)
							if (lamp.object == object)
							{
								colour = (uint32_t)(255.f * lamp.flame) | (uint32_t)(150.f * lamp.flame) << 8 | (uint32_t)(50.f * lamp.flame) << 16;
								break;
							}
						const float wide = (float)(ramHalf(object + 16) >> 5), tall = (float)(ramHalf(object + 18) >> 4 >> 1);
						const float dx = std::fabs(px - cameraX), dz = std::fabs(pz - cameraZ);
						const float angle = ((dx + dz - std::min(dx, dz) / 2) / 2 - (py - cameraY)) * (6.2831853f / 4096.f), cs = std::cos(angle), sn = std::sin(angle);
						static constexpr float Across[4] = { -1, 1, -1, 1 }, Down[4] = { -1, -1, 1, 1 };
						const uint32_t w24 = ramWord(object + 24), w28 = ramWord(object + 28);
						const uint16_t uvs[4] = { (uint16_t)w24, (uint16_t)w28, (uint16_t)ramHalf(object + 32), (uint16_t)ramHalf(object + 34) };
						DeadstationWorldQuad quad {};
						for (int j = 0; j < 4; j++)
						{
							// (flat to the camera: along its own right and down, in the world)
							const float ax = Across[j] * wide, ay = Down[j] * tall, sx2 = cs * ax - sn * ay, sy2 = sn * ax + cs * ay;
							quad.p[j][0] = (int16_t)(px + r11 * sx2 + r21 * sy2), quad.p[j][1] = (int16_t)(py + r12 * sx2 + r22 * sy2), quad.p[j][2] = (int16_t)(pz + r13 * sx2 + r23 * sy2);
							quad.uv[j][0] = (uint8_t)uvs[j], quad.uv[j][1] = (uint8_t)(uvs[j] >> 8);
							quad.rgb[j][0] = (uint8_t)colour, quad.rgb[j][1] = (uint8_t)(colour >> 8), quad.rgb[j][2] = (uint8_t)(colour >> 16);
						}
						// (added to what's there, as the game has it)
						quad.page = (uint16_t)(((w28 >> 16) & 0x1FF & ~0x60u) | 0x20u), quad.clut = (uint16_t)(w24 >> 16), quad.semi = 1 + 1;
						trackQuadsBlended.push_back(quad);
						trackQuadsBlendedZ.push_back(vz);
						continue;
					}
					// a tyre track: on the ground, in the world, before any car (the game's, 0x800129A0, in the ordering table, was painted over whatever was there)
					if (handler == TyreTrack && fullDetail)
					{
						// (its light from the ground it lies on, at each end)
						cpu->gpr[4] = (uint32_t)(int32_t)ramHalf(object + 4), cpu->gpr[5] = (uint32_t)(int32_t)ramHalf(object + 8);
						psx_dispatch_call(cpu, 0x8002BDEC, ra);
						const uint8_t nearShade = (uint8_t)((cpu->gpr[2] & 15) << 4);
						cpu->gpr[4] = (uint32_t)(int32_t)ramHalf(object + 36), cpu->gpr[5] = (uint32_t)(int32_t)ramHalf(object + 40);
						psx_dispatch_call(cpu, 0x8002BDEC, ra);
						const uint8_t farShade = (uint8_t)((cpu->gpr[2] & 15) << 4);
						cpu->gpr[29] = sp;
						const uint32_t w24 = ramWord(object + 24), w28 = ramWord(object + 28);
						const uint16_t uvs[4] = { (uint16_t)w24, (uint16_t)w28, (uint16_t)ramHalf(object + 32), (uint16_t)ramHalf(object + 34) };
						static constexpr uint32_t CornerAt[4] = { 48, 56, 64, 72 };
						DeadstationWorldQuad quad {};
						for (int j = 0; j < 4; j++)
						{
							quad.p[j][0] = ramHalf(object + CornerAt[j]), quad.p[j][1] = ramHalf(object + CornerAt[j] + 2), quad.p[j][2] = ramHalf(object + CornerAt[j] + 4);
							quad.uv[j][0] = (uint8_t)uvs[j], quad.uv[j][1] = (uint8_t)(uvs[j] >> 8);
							const uint8_t shade = j < 2 ? nearShade : farShade;
							quad.rgb[j][0] = quad.rgb[j][1] = quad.rgb[j][2] = shade;
						}
						quad.page = (uint16_t)((w28 >> 16) & 0x1FF), quad.clut = (uint16_t)(w24 >> 16), quad.layer = -2;
						trackMarks.push_back(quad);
						continue;
					}
					// (what a place sounds like, when it's drawn: not from everywhere, and at its own 30 a second)
					if ((handler == TrackSoundA || handler == TrackSoundB) && fullDetail)
					{
						const float dx = (float)ramHalf(object + 4) - cameraX, dz = (float)ramHalf(object + 8) - cameraZ;
						if (!tickPass || dx * dx + dz * dz > 15572.f * 15572.f)
							continue;
					}
					cpu->gpr[4] = ot, cpu->gpr[5] = cursorAt, cpu->gpr[6] = bufferEnd, cpu->gpr[7] = object;
					psx_dispatch_call(cpu, handler, ra);
					cpu->gpr[29] = sp;
				}
			// its polygons: in the mesh. DEADSTATION_SDK_OFF=drawin: the game's rings, one by one
			if (!fullDetail)
			{
			const uint32_t list = visit.list;
			uint32_t count = ramByte(tile + 76 + list);
			uint32_t poly = ramWord(tile + 80 + list * 4);
			for (; count != 0; count--, poly += 36)
			{
				const uint32_t code = ramByte(poly + 31);
				const bool quad = code == 0x3C || code == 0x2C, smooth = code == 0x3C || code == 0x34;
				TrackCorner q[4];
				float world[4][3];
				bool behind = false;
				for (int j = 0; j < (quad ? 4 : 3); j++)
				{
					const uint32_t v = poly + (uint32_t)j * 8;
					const float px = ramHalf(v), py = ramHalf(v + 2), pz = ramHalf(v + 4);
					world[j][0] = px, world[j][1] = py, world[j][2] = pz;
					const float vz = r31 * px + r32 * py + r33 * pz + tz;
					if (vz < 1.f)
					{
						behind = true;
						break;
					}
					const float s = h / vz;
					q[j].sx = ofx + (r11 * px + r12 * py + r13 * pz + tx) * s, q[j].sy = ofy + (r21 * px + r22 * py + r23 * pz + ty) * s, q[j].z = vz;
				}
				if (behind)
					continue;
				// the game's limit and facing: the third corner's depth; two-sided by byte 6
				int32_t order = (int32_t)((uint32_t)q[2].z >> 4);
				if (order <= 0)
					continue;
				// (a quad's two triangles wind opposite ways: the game's test, 0x80011BCC: its back is the first turned away and the second turned towards. A triangle by its one)
				const auto faces = [](const TrackCorner& a, const TrackCorner& b, const TrackCorner& d) {
					return (b.sx - a.sx) * (d.sy - a.sy) - (d.sx - a.sx) * (b.sy - a.sy) > 0;
				};
				if (ramByte(poly + 6) == 0 && !faces(q[0], q[1], q[2]) && (!quad || faces(q[1], q[2], q[3])))
					continue;
				if (visit.far)
					order += quad ? 48 : 16;
				if (order >= (fullDetail ? 4000 : 1504))
					continue;
				const uint32_t tex = ramWord(poly + 32) + texOffset;
				// (a texture record may lie in the scratchpad: the game fills some there as it goes)
				const bool inScratch = (tex >> 24) == 0x1F;
				const uint32_t w0 = inScratch ? rw(tex) : ramWord(tex), w1 = inScratch ? rw(tex + 4) : ramWord(tex + 4), w2 = inScratch ? rw(tex + 8) : ramWord(tex + 8);
				q[0].uv = (uint16_t)w0, q[1].uv = (uint16_t)w1, q[2].uv = (uint16_t)w2, q[3].uv = (uint16_t)(w2 >> 16);
				const uint32_t c0 = ramByte(poly + 22), c1 = ramByte(poly + 23), c2 = ramByte(poly + 30), c3 = ramByte(poly + 7);
				if (smooth)
				{
					q[0].rgb = c0 * 0x010101u, q[1].rgb = c1 * 0x010101u, q[2].rgb = c2 * 0x010101u;
					if (quad)
						q[3].rgb = c3 * 0x010101u;
				}
				else
					for (int j = 0; j < 4; j++)
						q[j].rgb = c0 | c1 << 8 | c2 << 16;
				const int32_t lying = (int8_t)ramByte(poly + 14);
				const uint16_t entry = (uint16_t)std::clamp<int32_t>(order + lying, 0, TrackEntryMost);
				// (its place in the game's order, +14 entries of 16 depth each, as a depth: what the game laid over it stays over it. What's laid on another lies within
				// about 64 of it (the tracks' data); more would let a far polygon show through a cliff in front of it)
				const int8_t layer = (int8_t)std::clamp<int32_t>(lying / 2, 0, 3);
				const uint16_t clut = (uint16_t)(w0 >> 16), page = (uint16_t)((w1 >> 16) & 0x1FF);
				const uint8_t kind = (uint8_t)((code & 1) | (code & 2));
				std::vector<TrackTriangle>& to = (kind & 2) != 0 ? trackBlended : trackTriangles;
				to.push_back({ { q[0], q[1], q[2] }, page, clut, entry, kind, layer });
				if (quad)
					to.push_back({ { q[1], q[3], q[2] }, page, clut, entry, kind, layer });
				if ((kind & 2) == 0)
					trackFarthest = std::max(trackFarthest, entry);
				// lit by a beam (only what's solid: a beam on a blended polygon would show what's behind it twice)
				if (beams != 0 && (kind & 2) == 0)
				drawn++;
			}
			}
		}
	}
	inTrack = false;
	buildingWorld.push_back({ cursor & 0x1FFFFF, psx_mod_read_word(cursorAt) & 0x1FFFFF });
	for (int i = 0; i < 8; i++)
		cpu->gte_ctrl[i] = saved[i];
	cpu->gpr[29] = sp0, cpu->gpr[31] = ra;
	cpu->gpr[2] = drawn;
	return 1;
}

// a frame's triangles by their table entries (a counting sort: the entries are small, the triangles many)
void sortByEntry(std::vector<TrackTriangle>& tris, bool farthestFirst)
{
	static std::vector<TrackTriangle> sorted;
	static uint32_t starts[0x801];
	if (tris.size() < 2)
		return;
	std::fill(std::begin(starts), std::end(starts), 0);
	for (const TrackTriangle& t : tris)
		starts[(farthestFirst ? 0x7FF - t.entry : t.entry) + 1u]++;
	for (unsigned i = 1; i <= 0x800; i++)
		starts[i] += starts[i - 1];
	sorted.resize(tris.size());
	for (const TrackTriangle& t : tris)
		sorted[starts[farthestFirst ? 0x7FF - t.entry : t.entry]++] = t;
	tris.swap(sorted);
}

// to the render thread in one block
void emitTrack(const TrackTriangle *tris, size_t count)
{
	if (count == 0)
		return;
	int32_t ox = 0, oy = 0;
	psx_gpu_draw_offset(&ox, &oy);
	const float fx = (float)ox, fy = (float)oy;
	static std::vector<DeadstationTriangle> out;
	out.resize(count);
	for (size_t i = 0; i < count; i++)
	{
		const TrackTriangle& t = tris[i];
		DeadstationTriangle& o = out[i];
		for (int j = 0; j < 3; j++)
		{
			o.x[j] = t.c[j].sx + fx, o.y[j] = t.c[j].sy + fy, o.z[j] = std::min(t.c[j].z, 65535.f);
			o.rgb[j] = t.c[j].rgb, o.uv[j] = t.c[j].uv;
		}
		o.page = t.page, o.clut = t.clut, o.raw = t.kind & 1, o.layer = t.layer;
		o.semi = (t.kind & 2) != 0 ? (uint8_t)(1 + ((t.page >> 5) & 3)) : 0;
	}
	deadstation_replay_object(0);
	currentObject = 0;
	if (tintWanted)
		setTint(true);
	deadstation_triangles(out.data(), count);
}

// the walk of the table reaches an entry: the opaque polygons at the first (the depth test orders them; things without depth
// lie as before: farther off behind them, nearer over them), the blended at theirs
void onListNode(uint32_t addr)
{
	if (!trackActive)
		return;
	const uint32_t at = addr & 0x1FFFFC, base = trackOt & 0x1FFFFC;
	if (at < base || at >= base + 0x2000)
		return;
	const uint32_t entry = (at - base) >> 2;
	if (!trackSorted)
	{
		sortByEntry(trackBlended, true);	// (the opaque ones in any order: the depth test and the GPU's hidden surface removal)
		trackSorted = true;
		trackEmitted = 0;
		trackOpaqueDrawn = false;
	}
	if (!trackOpaqueDrawn && entry <= trackFarthest)
	{
		trackOpaqueDrawn = true;
		addStars();
		emitTrack(trackTriangles.data(), trackTriangles.size());
		emitCarsAndLight();
		currentObject = 0;
	}
	size_t end = trackEmitted;
	while (end < trackBlended.size() && trackBlended[end].entry >= entry)
		end++;
	if (end != trackEmitted)
		emitTrack(trackBlended.data() + trackEmitted, end - trackEmitted);
	trackEmitted = end;
}

// ---- 60 frames of motion from its 30 (sdk/brick/replay.h) ----
//
// The race draws every second blank (VSync(2) at 0x80017C64) and its physics step once a frame, so its logic stays at 30.
// The blank between shows the frame's own polygons half a step on, on the render thread: the world by the camera's motion,
// each car by its own (its body's matrix). The car renderer (0x80012FF8: ordering table entry, the packet cursor's
// address, the buffer's end, the car, on the stack the camera) gives both, and the packets each car took: its polygons are
// tagged by their packets' addresses as the display list reaches them. In frames with no car (menus) nothing is replayed.
// Polygons with a traced depth in those frames are perspective correct as well. DEADSTATION_SDK_OFF=replay: the frame shown
// twice.
constexpr uint32_t CarRenderer = 0x80012FF8;
struct PacketRange
{
	uint32_t from, to;
	int object;
};
deadstation::replay::Frame building {};
std::vector<PacketRange> buildingRanges, drawingRanges;
uint32_t carIds[deadstation::replay::MaxObjects] = {};
bool inCar = false;

void toTransform(uint32_t matrix, deadstation::replay::Transform& out)
{
	double r[3][3], t[3];
	readMatrix(matrix, r, t);
	for (int i = 0; i < 3; i++)
	{
		for (int j = 0; j < 3; j++)
			out.r[3 * i + j] = (float)r[i][j];
		out.t[i] = (float)t[i];
	}
	out.present = true;
}
int onCarRenderer(CPUState *cpu, uint32_t)
{
	// (a split screen that isn't ours keeps the game's own: its frames aren't drawn by emitFrame)
	if (inCar || (guestByte(Players) > 1 && !splitNative))
		return 0;
	const uint32_t ra = cpu->gpr[31], cursor = cpu->gpr[5], car = cpu->gpr[7], camera = rw(cpu->gpr[29] + 16);
	const uint32_t from = rw(cursor);
	if (!building.objects[0].present)
	{
		toTransform(camera, building.objects[0]);
		building.ofx = (int32_t)cpu->gte_ctrl[24] / 65536.0f, building.ofy = (int32_t)cpu->gte_ctrl[25] / 65536.0f;
		building.h = (float)(uint16_t)cpu->gte_ctrl[26];
	}
	// this car's object: the same one from frame to frame
	unsigned id = 0;
	for (unsigned i = 1; i < deadstation::replay::MaxObjects && id == 0; i++)
		if (carIds[i] == car || carIds[i] == 0)
		{
			carIds[i] = car;
			id = i;
		}
	currentCarId = (int)id;
	inCar = true;
	{
		Timed timed(passTimes.cars);
		psx_dispatch_call(cpu, CarRenderer, ra);
	}
	inCar = false;
	cpu->gpr[31] = ra;
	currentCarId = 0;
	if (id != 0)
	{
		toTransform(Scratch + 32 * 4, building.objects[id]);	// its body's matrix, as the renderer left it
		buildingRanges.push_back({ from & 0x1FFFFF, rw(cursor) & 0x1FFFFF, (int)id });
	}
	return 1;
}
// ---- A car's polygons natively: 0x80012C74(the parts in the order for the camera, ordering table entry, packet cursor, buffer's end; on the
// stack, the car and its shade) as host code ----
//
// The game's renderer goes on working out the matrices (suspension, steering, damage) and leaves them in the scratchpad, this as they're
// ready: each part's polygons (44-byte records) projected with its matrix, with their true depths, each corner lit: the shade the game gives
// it in the race's light, plus the beams of the other cars that reach it, by the way the polygon faces them. No GTE and no packets, so no
// ordering table: the depth test orders them, and a light can know what's in front of it.
constexpr uint32_t CarPolygonsDraw = 0x80012C74;
// (the wheel at this place on the ground: the car's nearest of its four wheel points, whose first flag says so)
float wheelRise[8][4] = {};
inline bool wheelOnGround(uint32_t record, float x, float z)
{
	float nearest = 1e12f;
	bool down = false;
	for (uint32_t i = 0; i < 4; i++)
	{
		const uint32_t point = record + 476 + i * 192;
		const float dx = (float)(int32_t)rw(point) / 4096.f - x, dz = (float)(int32_t)rw(point + 8) / 4096.f - z, d = dx * dx + dz * dz;
		if (d < nearest)
			nearest = d, down = (psx_mod_read_half(record + 572 + i * 192) & 1) != 0;
	}
	return down;
}
int onCarPolygons(CPUState *cpu, uint32_t)
{
	if (guestByte(Players) > 1 && !splitNative)
		return 0;
	const uint32_t order = cpu->gpr[4], sp = cpu->gpr[29];
	const uint32_t car = rw(sp + 16), shadeValue = rw(sp + 20);
	const uint32_t parts = rw(car + CarParts) + 4, polygons = rw(car + CarPolygons);
	const float ofx = (int32_t)cpu->gte_ctrl[24] / 65536.0f, ofy = (int32_t)cpu->gte_ctrl[25] / 65536.0f, h = (float)(uint16_t)cpu->gte_ctrl[26];
	if (h <= 0)
		return 1;
	// the five matrices, once each
	struct PartMatrix
	{
		bool read;
		float r[9], t[3];
	} matrices[5] = {};
	static constexpr uint32_t MatrixAt[5] = { 64, 0, 96, 32, 128 };
	const uint32_t base = (shadeValue & 0xFF) != 0 ? shadeValue & 0xFF : 128;	// (no shade: raw, as it is)
	// the beams that can reach this car, not its own
	const CarLight *lights[8];
	int lightCount = 0;
	const uint32_t carPlace = rw(car + 100) + 176 + 20;
	const float carX = (float)(int32_t)rw(carPlace), carY = (float)(int32_t)rw(carPlace + 4), carZ = (float)(int32_t)rw(carPlace + 8);
	for (int i = 0; i < carLightCount; i++)
		if (carLights[i].object != car && (carX + 300 >= carLights[i].x0 && carX - 300 <= carLights[i].x1 && carZ + 300 >= carLights[i].z0 && carZ - 300 <= carLights[i].z1))
			lights[lightCount++] = &carLights[i];
	const float bounce = carLightCount > 0 ? 0.14f * look.beam : 0.f;	// (its own, thrown back off the ground)
	const float lampNear = lampLightAt(carX, carY, carZ);
	const float ambientR = std::min(look.world.r + bounce + lampNear * 1.1f, 2.f), ambientG = std::min(look.world.g + bounce * 0.94f + lampNear * 0.5f, 2.f), ambientB = std::min(look.world.b + bounce * 0.78f + lampNear * 0.12f, 2.f);
	const auto channel = [base](float k) { return (uint32_t)std::min((int)((float)base * k), 255); };
	const uint32_t plain = channel(ambientR) | channel(ambientG) << 8 | channel(ambientB) << 16;
	// (a car's decals, its plate and its stripes, lie on its panels or a hair inside them, and show by being drawn later: so among themselves a
	// car's polygons are in the game's order, as the PlayStation has them, against the world's depth; then their depth is written)
	carViewZ[currentCarId] = (float)(int32_t)rw(Scratch + 128 + 28);	// (its body's matrix, its place's depth)
	std::vector<DeadstationTriangle>& solid = carSolid[currentCarId], &blended = carBlended[currentCarId];
	std::vector<DeadstationGlow>& depths = carDepths[currentCarId];
	const uint32_t count = rb(order);
	for (uint32_t i = 0; i < count; i++)
	{
		const uint32_t index = rb(order + 1 + i), part = parts + index * 44;
		const uint32_t lod = ramByte(part + 24);
		if (ramByte(part + 25) == 0)
			continue;
		const uint32_t polygonCount = (uint16_t)ramHalf(part + lod * 2 + 16), first = (uint16_t)ramHalf(part + lod * 2 + 20);
		if (polygonCount == 0)
			continue;
		PartMatrix& m = matrices[index >= 12 ? 4 : (index & 3) == 0 ? 0 : (index & 3) == 1 ? 1 : (index & 3) == 2 ? 2 : 3];
		if (!m.read)
		{
			m.read = true;
			const uint32_t at = Scratch + MatrixAt[index >= 12 ? 4 : (index & 3) == 0 ? 0 : (index & 3) == 1 ? 1 : (index & 3) == 2 ? 2 : 3];
			for (int k = 0; k < 9; k++)
				m.r[k] = (float)rh(at + (uint32_t)k * 2) / 4096.f;
			for (int k = 0; k < 3; k++)
				m.t[k] = (float)(int32_t)rw(at + 20 + (uint32_t)k * 4);
			// A wheel on what's drawn: it stands on the physics' ground, and the polygons drawn there can be a little above or below that. By the difference, within reason, and only
			// while it's on the ground: what's under a wheel in the air means nothing to it, and that difference changes by the grid square. It follows smoothly, either way.
			if (index < 12)
			{
				const int corner = (int)(index & 3);
				float target = 0;
				const float dx = m.t[0] - trackCamera.t[0], dy = m.t[1] - trackCamera.t[1], dz = m.t[2] - trackCamera.t[2];
				const float wx = trackCamera.r[0] * dx + trackCamera.r[3] * dy + trackCamera.r[6] * dz, wz = trackCamera.r[2] * dx + trackCamera.r[5] * dy + trackCamera.r[8] * dz;
				if (!meshFloor.empty() && wheelOnGround(rw(car + 100), wx, wz))
				{
					const int16_t floor = surfaceAt(meshFloor, wx, wz);
					if (floor != SurfaceNone)
					{
						const uint32_t ra = cpu->gpr[31], a0 = cpu->gpr[4], a1 = cpu->gpr[5];
						Ground ground(cpu);
						target = std::clamp((float)floor - (float)ground.height((int)wx, (int)wz), -32.f, 32.f);
						cpu->gpr[31] = ra, cpu->gpr[4] = a0, cpu->gpr[5] = a1;
					}
				}
				float& rise = wheelRise[currentCarId][corner];
				rise += (target - rise) * 0.2f;
				m.t[0] += trackCamera.r[1] * rise, m.t[1] += trackCamera.r[4] * rise, m.t[2] += trackCamera.r[7] * rise;
			}
		}
		for (uint32_t q = 0; q < polygonCount; q++)
		{
			const uint32_t poly = polygons + (first + q) * 44;
			const bool quad = ramHalf(poly + 18) != 3;
			const int corners = quad ? 4 : 3;
			float sx[4], sy[4], vz[4], vx[4], vy[4];
			bool behind = false;
			for (int j = 0; j < corners; j++)
			{
				const uint32_t at = poly + 12 + (uint32_t)j * 8;
				const float x = ramHalf(at), y = ramHalf(at + 2), z = ramHalf(at + 4);
				vx[j] = m.r[0] * x + m.r[1] * y + m.r[2] * z + m.t[0];
				vy[j] = m.r[3] * x + m.r[4] * y + m.r[5] * z + m.t[1];
				vz[j] = m.r[6] * x + m.r[7] * y + m.r[8] * z + m.t[2];
				behind = behind || vz[j] < 8.f;
				if (behind)
					break;
				const float k = h / vz[j];
				sx[j] = ofx + vx[j] * k, sy[j] = ofy + vy[j] * k;
			}
			if (behind)
				continue;
			const auto facing = [&](int a, int b, int c) { return (sx[b] - sx[a]) * (sy[c] - sy[a]) - (sx[c] - sx[a]) * (sy[b] - sy[a]); };
			const bool twoSided = ramByte(poly + 1) != 0;
			const float front = facing(0, 1, 2);
			if (!twoSided && !(front > 0 || (quad && facing(1, 2, 3) <= 0)))
				continue;
			const uint32_t w4 = ramWord(poly + 4), w8 = ramWord(poly + 8);
			const uint16_t uv[4] = { (uint16_t)w4, (uint16_t)w8, (uint16_t)ramHalf(poly + 34), (uint16_t)ramHalf(poly + 42) };
			const uint16_t page = (uint16_t)((w8 >> 16) & 0x1FF), clut = (uint16_t)(w4 >> 16);
			uint32_t rgb[4] = { plain, plain, plain, plain };
			if (lightCount > 0 && eye.valid)
			{
				// each corner in the world (the camera's inverse is its transpose)
				float wx[4], wy[4], wz[4];
				for (int j = 0; j < corners; j++)
				{
					const float dx = vx[j] - eye.t[0], dy = vy[j] - eye.t[1], dz = vz[j] - eye.t[2];
					wx[j] = eye.r[0] * dx + eye.r[3] * dy + eye.r[6] * dz;
					wy[j] = eye.r[1] * dx + eye.r[4] * dy + eye.r[7] * dz;
					wz[j] = eye.r[2] * dx + eye.r[5] * dy + eye.r[8] * dz;
				}
				const float ax = wx[1] - wx[0], ay = wy[1] - wy[0], az = wz[1] - wz[0], bx = wx[2] - wx[0], by = wy[2] - wy[0], bz = wz[2] - wz[0];
				float nx = ay * bz - az * by, ny = az * bx - ax * bz, nz = ax * by - ay * bx;
				const float nl = std::max(std::sqrt(nx * nx + ny * ny + nz * nz), 0.001f);
				nx /= nl, ny /= nl, nz /= nl;
				for (int j = 0; j < corners; j++)
				{
					float lit = 0;
					for (int l = 0; l < lightCount; l++)
					{
						const CarLight& c = *lights[l];
						const float strength = beamAt(c, wx[j], wy[j], wz[j]);
						if (strength <= 0)
							continue;
						// (the models wind their polygons so that this is positive where the light falls on them, as the shadows use it)
						const float lx = wx[j] - c.x, ly = wy[j] - c.y, lz = wz[j] - c.z, ll = std::max(std::sqrt(lx * lx + ly * ly + lz * lz), 1.f);
						const float faces = std::fabs(ramByte(poly + 1) != 0 ? 1.f : (lx * nx + ly * ny + lz * nz) / ll);
						lit += strength * (0.35f + 1.1f * faces);
					}
					lit = std::min(lit * look.beam * 1.9f, 1.8f);
					rgb[j] = channel(ambientR + lit) | channel(ambientG + lit * 0.94f) << 8 | channel(ambientB + lit * 0.78f) << 16;
				}
			}
			const bool see = (rb(poly + 27) & 2) != 0;
			const uint8_t semi = see ? (uint8_t)(1 + ((page >> 5) & 3)) : 0;
			const auto add = [&](int a, int b, int c) {
				DeadstationTriangle t {};
				const int k[3] = { a, b, c };
				for (int j = 0; j < 3; j++)
				{
					const int v = k[j];
					t.x[j] = sx[v], t.y[j] = sy[v], t.z[j] = std::clamp(vz[v], 16.f, 65535.f), t.rgb[j] = rgb[v], t.uv[j] = uv[v];
				}
				t.page = page, t.clut = clut, t.raw = 0, t.semi = see ? semi : DeadstationPainted, t.layer = CarInFront;
				(see ? blended : solid).push_back(t);
				if (!see)
					depths.push_back({ { t.x[0], t.x[1], t.x[2] }, { t.y[0], t.y[1], t.y[2] }, { 0, 0, 0 },
							{ std::max(t.z[0] - CarDepthNearer, 16.f), std::max(t.z[1] - CarDepthNearer, 16.f), std::max(t.z[2] - CarDepthNearer, 16.f) } });
			};
			add(0, 1, 2);
			if (quad)
				add(1, 3, 2);
		}
	}
	return 1;
}


// ---- 60 frames of the game a second ----
//
// In a one-player race the game's whole loop (0x80020074) goes round each blank. What flows in time runs on every pass, at half its step per pass:
// - the physics: every integration multiplies by 0x2222 at shift 18, 1/30 of a second, through one helper (0x8003F48C), a wheel's turn by 0x56F, and
//   each car keeps its step over its mass from when it's set up (+640, +644);
// - the player's controls (0x8002DF3C): steering, throttle and brake in steps per pass, the throttle's rates in the game's own variables;
// - the AI's hands (0x8003BB40: steering, throttle, brake towards what it wants). Its eyes (0x8003E9F0: speed as the distance between two looks, where it
//   is on the route) and its decisions keep their pace, as what it was tuned for, with the rest of 0x8003B3EC, on every second pass;
// - the camera (0x8002D3F0; on a string behind the car, so only its swing and its rise are rates), the frame.
// What's counted keeps real time, on every second pass (tickPass): the race clock (it divides by 30 to show it), the sounds, the texture animations, the
// fades, what cars throw up (0x800373C0, whose bits of grit and smoke fly half a step on between, on what the blank between draws). Laps are watched
// every pass: a car's last move crosses the line on one or the other.
//
// A fixed step that misses a frame is slow motion, so the time is kept here, by the clock. A pass that presents a frame can't be shorter than a panel's refresh (the
// render thread's queue holds it), so a frame that overran can't be made up by more frames: a whole frame behind, the next pass is a step of the game alone, with nothing
// to show (a third of the cost of a drawn one, all of the game's time). So real frames as many as the Brick can draw, and the game's speed always. Split screens' loops
// are the game's. DEADSTATION_SDK_OFF=sixty: the game's 30.
constexpr uint32_t PhysicsStep = 0x80030ACC, Multiply = 0x8003F48C, PadsRead = 0x8002CBE8, RaceBookkeeping = 0x80043914, DrawingFlag = 0x80060AC4;
constexpr uint32_t PadStates = 0x800B7E9C, Gp = 0x800608AC, BlankCount = 0x80058D78, ReplayMode = 0x80060DF0, ReplayOrbit = 0x80060BD4;
// (each one's return, in the one-player loop 0x80020074)
constexpr uint32_t PhysicsBack = 0x800204A0, PadsBack = 0x80020438, BookkeepingBack = 0x80020970;
constexpr uint32_t WheelSpinA = 0x8003413C, WheelSpinB = 0x80034190;	// (the returns of its two multiplies)
bool halfStep = false, inPhysics = false, inPads = false;
double physicsMs = 0;
inline bool sixtyNow(uint32_t ra, uint32_t back)
{
	return sixty && ra == back && guestByte(Players) <= 1;
}
inline int16_t &guestHalf(uint32_t a) { return *(int16_t *)(g_psx_ram + (a & 0x1FFFFE)); }
// The helper everything in the physics multiplies with, a * b >> shift, rounded (0x8003F48C), native. And the time step, which it's handed
constexpr uint32_t TyreStateA = 0x80032EA0, TyreStateB = 0x8003394C, TyreBody = 0x80033310;
bool helpersNative = true;	// DEADSTATION_SDK_OFF=helpers: the game's own; the AI's race is the same to the unit, or these are wrong
int onMultiply(CPUState *cpu, uint32_t)
{
	uint32_t b = cpu->gpr[5];
	const uint32_t shift = cpu->gpr[6];
	// A wheel's spin keeps the game's own rate: a full step on every second pass, nothing on the one between. It's a stiff thing, little inertia against a lot of torque, so how often it's stepped
	// changes how it behaves and not just how fast: at half a step a stopped car rolls on a slope where at 30 it stands, a full step each pass and the revs and the gearbox go wrong, and either way the
	// AI's line differs. The body and the rest step at 60, with the wheels' last spin held for the pass between, as the game's own step would have it.
	if (halfStep && shift == 18)
	{
		if (cpu->gpr[31] == WheelSpinA || cpu->gpr[31] == WheelSpinB)
			b = tickPass ? b : 0;
		// (what a tyre works out its grip from, wheel and car, is over the wheel's own step: a whole one. At half of it a tyre thought the same slip wanted twice the force, and a car
		// went 7% too fast at 6 seconds from the line, and shifted too soon)
		else if (cpu->gpr[31] == TyreBody)
			b *= 2;
		else if (cpu->gpr[31] != TyreStateA && cpu->gpr[31] != TyreStateB)
			b = b == 0x2222 ? 0x1111 : b == 0x56F ? 0x2B8 : b;
	}
	if (!helpersNative)
	{
		cpu->gpr[5] = b;
		return 0;
	}
	const int64_t full = (int64_t)(int32_t)cpu->gpr[4] * (int64_t)(int32_t)b;
	const uint32_t lo = (uint32_t)full, hi = (uint32_t)((uint64_t)full >> 32);
	cpu->lo = lo, cpu->hi = hi;
	const uint32_t low = lo >> ((shift - 1) & 31);
	cpu->gpr[2] = ((low >> 1) | (hi << ((32 - shift) & 31))) + (low & 1);
	return 1;
}
// The ground's height. The game unpacks a row of its heights into the scratchpad for each look (0x8002C194), each wheel of each car each step and the particles too: about 15% of the thread,
// for what's the same all race. From the table instead, the same numbers: only those two functions read that row, and everyone else takes the answer in v0.
int onGridSample(CPUState *cpu, uint32_t)
{
	if (!helpersNative || Ground::filling)
		return 0;
	const uint32_t ra = cpu->gpr[31];
	Ground ground(cpu);
	const int h = ground.gridSample((int32_t)cpu->gpr[4], (int32_t)cpu->gpr[5]);
	cpu->gpr[31] = ra;
	cpu->gpr[2] = (uint32_t)h;
	return 1;
}
int onHeightNative(CPUState *cpu, uint32_t)
{
	onHeight(cpu, 0);	// (the shadows' look at it)
	if (!helpersNative)
		return 0;
	const uint32_t ra = cpu->gpr[31];
	Ground ground(cpu);
	const int h = ground.height((int32_t)cpu->gpr[4], (int32_t)cpu->gpr[5]);
	cpu->gpr[31] = ra;
	cpu->gpr[2] = (uint32_t)(int32_t)h;
	return 1;
}
// (three words, through memory that's as often the scratchpad, the physics keeping its stack there)
// a . b >> 12, rounded (0x8003F4DC)
int onDot(CPUState *cpu, uint32_t)
{
	const uint32_t a = cpu->gpr[4], b = cpu->gpr[5];
	const int64_t sum = (int64_t)(int32_t)rw(a) * (int32_t)rw(b) + (int64_t)(int32_t)rw(a + 4) * (int32_t)rw(b + 4) + (int64_t)(int32_t)rw(a + 8) * (int32_t)rw(b + 8);
	const uint32_t lo = (uint32_t)sum, hi = (uint32_t)((uint64_t)sum >> 32), low = lo >> 11;
	cpu->gpr[2] = ((low >> 1) | (hi << 20)) + (low & 1);
	return 1;
}
// a whole square root, by the game's own digits (0x8003F564)
int onRoot(CPUState *cpu, uint32_t)
{
	uint32_t value = cpu->gpr[4], root = 0, left = 0;
	for (int i = 0; i < 22; i++)
	{
		left |= value >> 30;
		value <<= 2;
		root <<= 1;
		const uint32_t next = (root << 1) + 1;
		if ((int32_t)left >= (int32_t)next)
		{
			left -= next;
			root += 1;
		}
		left <<= 2;
	}
	cpu->gpr[2] = root;
	return 1;
}
// a rough length: the longer plus half the shorter (0x8003CA54)
int onRoughLength(CPUState *cpu, uint32_t)
{
	int32_t a = (int32_t)cpu->gpr[4], b = (int32_t)cpu->gpr[5];
	a = a < 0 ? -a : a, b = b < 0 ? -b : b;
	cpu->gpr[2] = (uint32_t)(a + b - (((uint32_t)a < (uint32_t)b ? a : b) >> 1));
	return 1;
}
// LIBGTE's VectorNormal and VectorNormalS (0x8004B0BC, 0x8004B0E8, on 0x8004B118) without the GTE: the sum of squares through the game's table of inverse roots
inline uint32_t normalOf(int32_t x, int32_t y, int32_t z, int32_t out[3])
{
	const int16_t ix = (int16_t)x, iy = (int16_t)y, iz = (int16_t)z;
	const int32_t sum = (int32_t)((uint32_t)((int32_t)ix * ix) + (uint32_t)((int32_t)iy * iy) + (uint32_t)((int32_t)iz * iz));
	const uint32_t bits = sum >= 0 ? (uint32_t)sum : ~(uint32_t)sum;
	const int leading = (bits == 0 ? 32 : __builtin_clz(bits)) & ~1;
	const int shift = (31 - leading) >> 1;
	const int place = leading >= 24 ? (int32_t)((uint32_t)sum << (leading - 24)) : sum >> (24 - leading);
	const int16_t root = (int16_t)ramHalf(0x8005AE10u + (uint32_t)((place - 64) * 2));
	out[0] = ((int32_t)root * ix) >> (shift & 31), out[1] = ((int32_t)root * iy) >> (shift & 31), out[2] = ((int32_t)root * iz) >> (shift & 31);
	return (uint32_t)sum;
}
int onVectorNormal(CPUState *cpu, uint32_t)
{
	const uint32_t in = cpu->gpr[4], to = cpu->gpr[5];
	int32_t out[3];
	cpu->gpr[2] = normalOf((int32_t)rw(in), (int32_t)rw(in + 4), (int32_t)rw(in + 8), out);
	psx_mod_write_word(to, (uint32_t)out[0]), psx_mod_write_word(to + 4, (uint32_t)out[1]), psx_mod_write_word(to + 8, (uint32_t)out[2]);
	return 1;
}
int onVectorNormalS(CPUState *cpu, uint32_t)
{
	const uint32_t in = cpu->gpr[4], to = cpu->gpr[5];
	int32_t out[3];
	cpu->gpr[2] = normalOf((int16_t)psx_mod_read_half(in), (int16_t)psx_mod_read_half(in + 2), (int16_t)psx_mod_read_half(in + 4), out);
	psx_mod_write_half(to, (uint16_t)out[0]), psx_mod_write_half(to + 2, (uint16_t)out[1]), psx_mod_write_half(to + 4, (uint16_t)out[2]);
	return 1;
}
// DEADSTATION_RALLYCROSS_FREEZE=N (tests): N passes into a race the cars stop where they are, the camera with them, all else going on. What changes between two frames then is what flashes.
uint32_t freezeAfter = 0, racePasses = 0;
// (tests: the same hands on the same car at either rate, below)
bool handsTest = false;
void handsBeforePhysics();
void handsAfterPhysics();
void handsBeforeBrains();
int onPhysicsStep(CPUState *cpu, uint32_t)
{
	const uint32_t ra = cpu->gpr[31];
	if (handsTest && !inPhysics && ra == PhysicsBack)
	{
		handsBeforePhysics();
		const bool wasPhysics = inPhysics, half = sixty && guestByte(Players) <= 1;
		inPhysics = true, halfStep = half;
		psx_dispatch_call(cpu, PhysicsStep, ra);
		inPhysics = wasPhysics, halfStep = false;
		cpu->gpr[31] = ra;
		handsAfterPhysics();
		return 1;
	}
	if (inPhysics || !sixtyNow(ra, PhysicsBack))
		return 0;
	if (freezeAfter != 0 && ++racePasses > freezeAfter)
	{
		if (costing && (racePasses - freezeAfter) % 480 == 0)
			costNext();
		return 1;
	}
	inPhysics = halfStep = true;
	{
		Timed t(physicsMs);
		psx_dispatch_call(cpu, PhysicsStep, ra);
	}
	inPhysics = halfStep = false;
	cpu->gpr[31] = ra;
	return 1;
}

// A rate in steps, at half: an odd step goes on the first pass of a run, so a press answers at once
struct HalfRate
{
	int32_t owed = 0;
	int sign = 0;
	int move(int delta)
	{
		if (delta == 0)
		{
			owed = 0, sign = 0;
			return 0;
		}
		const int s = delta > 0 ? 1 : -1;
		if (s != sign)
			owed = 0, sign = s;
		owed += delta;
		const int m = (owed + s) / 2;
		owed -= m * 2;
		return m;
	}
};
struct CarControls
{
	HalfRate steering, throttle, brake;
};
CarControls carControls[8];
uint64_t passesKept = 0;
int passesUndrawn = 0;
constexpr int MostUndrawn = 3;
HalfRate cameraRise, cameraSwing;

bool autopilot = false;
constexpr uint32_t CarRecordsAt = 0x800B3448;
// the cars set up: each one's time step over its mass, and the inverse (+640, +644), worked out once here; and a race's start for all of this
constexpr uint32_t CarsSetUp = 0x8002ECD0;
// A turned over car's righting kick (0x800323DC). A press is one step of force: (its least + what it's risen to) x how far over the car is, the rise adding on with each kick, up to its most,
// and back to nothing with all four wheels down. A kick split over two passes loses by the second: the car's already moving and the ground and the springs have had a pass to answer, so it
// took about twice as many presses to get a car back as at the game's 30. So here too a kick is one step, on one pass. Twice the force would be the same push over half the time, but the half
// step's car answers less to it, so it's by the measure: flipped twelve times, pressed the same, 2.0 took 4.7 kicks, 2.6 took 3.2, the game's 30 3.3.
constexpr uint32_t KickLeast = Gp + 1284, KickRamp = Gp + 1244, KickMost = Gp + 1256;
constexpr int KickLeastAt30 = 0x270, KickRampAt30 = 50, KickMostAt30 = 0x400;
int kickPercent = 250;		// DEADSTATION_RALLYCROSS_KICK (tests)
constexpr uint32_t PointForceFlag = Gp + 1472;
bool inSetUp = false;
int onCarsSetUp(CPUState *cpu, uint32_t)
{
	if (inSetUp)
		return 0;
	const uint32_t ra = cpu->gpr[31];
	if (!sixty || guestByte(Players) > 1)
	{
		// (at the game's own rate, only for the test's driver)
		if (!autopilot)
			return 0;
		inSetUp = true;
		psx_dispatch_call(cpu, CarsSetUp, ra);
		inSetUp = false;
		cpu->gpr[31] = ra;
		for (uint32_t i = 0; i < guestByte(Players) || i == 0; i++)
			guestHalf(CarRecordsAt + i * 4100 + 3972) = 1;
		return 1;
	}
	inSetUp = halfStep = true;
	psx_dispatch_call(cpu, CarsSetUp, ra);
	inSetUp = halfStep = false;
	cpu->gpr[31] = ra;
	guestHalf(KickLeast) = (int16_t)(KickLeastAt30 * kickPercent / 100);
	guestHalf(KickRamp) = (int16_t)(KickRampAt30 * kickPercent / 100);
	guestHalf(KickMost) = (int16_t)(KickMostAt30 * kickPercent / 100);
	if (stats)
	{
		char table[256] = "";
		for (uint32_t i = 0; i < 16; i++)
			std::snprintf(table + std::strlen(table), sizeof(table) - std::strlen(table), " %d", (int32_t)guestRead(0x800B9700 + i * 4));
		std::fprintf(stderr, "deadstation rallycross: drag table%s\n", table);
	}
	// DEADSTATION_RALLYCROSS_AUTOPILOT (tests): the player's car driven by the game's AI, as its attract demo has it: the physics, the collisions and the track at racing speed
	if (autopilot)
		guestHalf(CarRecordsAt + 3972) = 1;
	tickPass = true;
	for (CarControls& c : carControls)
		c = CarControls {};
	cameraRise = cameraSwing = HalfRate {};
	racePasses = 0;
	keptFrom = Clock::time_point {};
	passesUndrawn = 0;
	return 1;
}
// (the test's driver: at the last lap the game gives the player's car back to its person's controls, 0x8003C4A0, for the last lap's state. With nobody on the pad it would coast to a stop. Not now)
int onLastLapHandover(CPUState *, uint32_t)
{
	return autopilot ? 1 : 0;
}
// what runs on every second pass
template <uint32_t Back> int everySecondPass(CPUState *cpu, uint32_t)
{
	return !tickPass && sixtyNow(cpu->gpr[31], Back) ? 1 : 0;
}

// The brains, 0x8003B3EC: the whole of them on the passes they count on; on those between, each car's hands only
constexpr uint32_t Brains = 0x8003B3EC, BrainsBack = 0x8002047C, CarRecords = 0x800B3448, CarRecordSize = 4100;
constexpr uint32_t HumanControls = 0x8002DF3C, AiBrain = 0x8003BA90, AiHands = 0x8003BB40;
// (each car's brain now, what it is if it's a person's or not, and the states of a person's that call the controls)
constexpr uint32_t BrainNow = 3924, BrainIsAi = 3972, Steering = 208, Throttle = 210, Brake = 212, GearMode = 214, GearHold = 215, Kick = 220, KickWait = 221, AiPush = 4084, AiPull = 4088;
inline bool personsControls(uint32_t brain)
{
	return brain == HumanControls || brain == 0x8003C704 || brain == 0x8003C774 || brain == 0x8003C874;
}
// At night the AI drives at half the speed it would by day: it can see no more than a person can. What it wants is in its route, on the point it's making for (+12, read only by its
// hands; what it adds for the difficulty is worked out from it), so its hands see half of it while they run: it brakes for corners and takes them as it does by day, only slower. The
// route itself stays as it was loaded, for a restart. (The autopilot's car stays a racing load to measure with.)
constexpr uint32_t RouteTable = 0x80065B3C, RouteAt = 3946, RoutePoint = 3962, PointSize = 16, PointSpeed = 12;
bool inHands = false;
int onAiHands(CPUState *cpu, uint32_t)
{
	const uint32_t car = cpu->gpr[4], ra = cpu->gpr[31];
	if (inHands || !nightRace || (autopilot && car == CarRecords))
		return 0;
	const uint32_t route = guestRead(RouteTable + (uint32_t)(int32_t)(int8_t)guestByte(car + RouteAt) * 4u);
	const uint32_t wanted = route + (uint32_t)(int32_t)guestHalf(car + RoutePoint) * PointSize + PointSpeed;
	const int16_t byDay = guestHalf(wanted);
	guestHalf(wanted) = (int16_t)(byDay / 2);
	inHands = true;
	psx_dispatch_call(cpu, AiHands, ra);
	inHands = false;
	guestHalf(wanted) = byDay;
	cpu->gpr[31] = ra;
	return 1;
}

bool inBrains = false;
int onBrains(CPUState *cpu, uint32_t)
{
	const uint32_t ra = cpu->gpr[31];
	if (handsTest && !inBrains && ra == BrainsBack)
		handsBeforeBrains();
	if (inBrains || !sixtyNow(ra, BrainsBack))
		return 0;
	inBrains = true;
	// the throttle's and the brake's rates, a person's: per step in the game's variables (0x8002F0xx), whole numbers kept whole over two passes
	guestWord(Gp + 1340, 1);
	guestWord(Gp + 1416, tickPass ? 10 : 9);
	guestWord(Gp + 1412, tickPass ? 48 : 47);
	guestWord(Gp + 1496, 26);
	const int cars = std::min<int>(guestByte(0x8005649A), 8);
	struct Was { int16_t steering, throttle, brake; uint8_t kickWait; int32_t push, pull; } was[8];
	for (int i = 0; i < cars; i++)
	{
		const uint32_t car = CarRecords + (uint32_t)i * CarRecordSize;
		was[i] = { guestHalf(car + Steering), guestHalf(car + Throttle), guestHalf(car + Brake), guestByte(car + KickWait),
				(int32_t)guestRead(car + AiPush), (int32_t)guestRead(car + AiPull) };
	}
	Timed timed(passTimes.brains);
	if (tickPass)
		psx_dispatch_call(cpu, Brains, ra);
	else
		for (int i = 0; i < cars; i++)
		{
			const uint32_t car = CarRecords + (uint32_t)i * CarRecordSize, brain = guestRead(car + BrainNow);
			const bool person = guestHalf(car + BrainIsAi) == 0;
			if (person ? personsControls(brain) : brain == AiBrain)
			{
				cpu->gpr[4] = car;
				// (a person's automatic gearbox goes by the revs, new only on the passes the wheels' spin is: on the pass between, it would shift again on the same revs, 2nd to 4th.
				// Its own test holds it off with this byte, and clears it)
				if (person && guestByte(car + GearMode) != 0)
					guestByte(car + GearHold) = 1;
				psx_dispatch_call(cpu, person ? HumanControls : AiHands, ra);
			}
		}
	for (int i = 0; i < cars; i++)
	{
		const uint32_t car = CarRecords + (uint32_t)i * CarRecordSize;
		CarControls& c = carControls[i];
		const auto halve = [](int16_t& now, int16_t then, HalfRate& rate) {
			const int delta = now - then;
			// (let go: at once)
			if (now == 0 && std::abs(delta) > 2)
			{
				rate = HalfRate {};
				return;
			}
			now = (int16_t)(then + rate.move(delta));
		};
		halve(guestHalf(car + Steering), was[i].steering, c.steering);
		halve(guestHalf(car + Throttle), was[i].throttle, c.throttle);
		halve(guestHalf(car + Brake), was[i].brake, c.brake);
		if (guestHalf(car + BrainIsAi) != 0)
		{
			// (its kick, as a person's, one pass: its brain sets it on the passes it thinks, and it would stay for the next)
			if (!tickPass)
				guestByte(car + Kick) = 0;
			// (what the AI's let go of, by the amount a step, and at once what it takes up)
			for (const auto [at, then] : { std::pair<uint32_t, int32_t>(AiPush, was[i].push), std::pair<uint32_t, int32_t>(AiPull, was[i].pull) })
			{
				const int32_t now = (int32_t)guestRead(car + at);
				if (now < then)
					guestWord(car + at, (uint32_t)(then - (then - now + 1) / 2));
			}
		}
		else
		{
			// a person's righting kick: 20 steps apart, so 40 passes
			if (!tickPass && guestByte(car + KickWait) + 1 == was[i].kickWait)
				guestByte(car + KickWait) = was[i].kickWait;
		}
	}
	// (what the force between a car's points is measured against, for its panels: raised as the kick's is, on the pass it's kicked, or a kick would knock them about)
	bool kicked = false;
	for (int i = 0; i < cars; i++)
		kicked = kicked || guestByte(CarRecords + (uint32_t)i * CarRecordSize + Kick) != 0;
	guestWord(PointForceFlag, kicked ? (uint32_t)((int64_t)0x400000 * kickPercent / 100) : 0x400000u);
	inBrains = false;
	cpu->gpr[31] = ra;
	return 1;
}


// ---- Tests: the same hands on the same car at either rate ----
//
// DEADSTATION_RALLYCROSS_DRIVE="from-to:mask,..." (seconds of the race, pad bits): a scripted drive, the car's place, speed and attitude every tenth of a second.
// DEADSTATION_RALLYCROSS_ROLL=N: N times the car is pushed over from the side, left to rest, and the triggers pressed by turns five times a second until it's back on its wheels. Each
// on the game's clock, so DEADSTATION_SDK_OFF=sixty (the game's own 30) and this are the same hands.
struct HandsPress { float from, to; uint16_t mask; };
std::vector<HandsPress> handsDrive;
int rollTrials = 0;
uint64_t handsSteps = 0;
inline uint32_t handsRate() { return sixty && guestByte(Players) <= 1 ? 60u : 30u; }
struct RollTest
{
	enum Phase { Waiting, Pushing, Resting, Pressing, Upright, Finished } phase = Waiting;
	int trial = 0, presses = 0, accepted = 0, forced = 0;
	uint32_t at = 0, startedAt = 0, contactSteps = 0, steps = 0, restSteps = 0;
	uint32_t contactA = 0, contactB = 0;
	bool next = false;
	int16_t restUp = 0;
} roll;
inline float carUpY() { return (float)guestHalf(CarRecords + 184) / 4096.f; }
inline bool wheelsDown()
{
	const uint32_t c = CarRecords;
	return (psx_mod_read_half(c + 572) & 1) && (psx_mod_read_half(c + 764) & 1) && (psx_mod_read_half(c + 956) & 1) && (psx_mod_read_half(c + 1148) & 1);
}
constexpr uint32_t RaceClock = 0x800659C0;
void handsBeforeBrains()
{
	if (guestRead(RaceClock) == 0)
		return;
	const float now = (float)handsSteps / (float)handsRate();
	uint16_t held = 0;
	for (const HandsPress& p : handsDrive)
		if (now >= p.from && now < p.to)
			held |= p.mask;
	uint16_t edges = 0;
	if (rollTrials > 0 && roll.phase == RollTest::Pressing)
	{
		const uint32_t every = handsRate() / 5;
		if ((handsSteps - roll.startedAt) % every == 0)
		{
			const uint16_t mask = roll.next ? 2 : 1;
			roll.next = !roll.next;
			held |= mask, edges |= mask;
			roll.presses++;
		}
	}
	static uint16_t heldWas = 0;
	edges |= held & ~heldWas;
	const uint16_t released = heldWas & ~held;
	heldWas = held;
	if (held != 0 || edges != 0 || released != 0)
	{
		const uint32_t pad = PadStates;
		psx_mod_write_half(pad, (uint16_t)(psx_mod_read_half(pad) | held));
		psx_mod_write_half(pad + 8, (uint16_t)(psx_mod_read_half(pad + 8) | edges));
		psx_mod_write_half(pad + 10, (uint16_t)(psx_mod_read_half(pad + 10) | released));
	}
}
void handsBeforePhysics()
{
	if (guestRead(RaceClock) == 0)
		return;
	handsSteps++;
	if (rollTrials == 0 || roll.phase != RollTest::Pushing)
		return;
	// (world up is y's negative: one side's points lifted, the other's held down, until it's past its side)
	const uint32_t c = CarRecords;
	const int side = roll.trial & 1;
	static constexpr int Lift[2][4] = { { 1, 2, 5, 6 }, { 0, 3, 4, 7 } };
	for (int i = 0; i < 4; i++)
	{
		const uint32_t f = c + 508 + (uint32_t)Lift[side][i] * 192 + 4;
		guestWord(f, guestRead(f) - 0x300000u);
	}
}
void handsAfterPhysics()
{
	if (guestRead(RaceClock) == 0)
		return;
	const uint32_t rate = handsRate(), c = CarRecords;
	if (handsSteps % (rate / 10) == 0 && !handsDrive.empty())
		std::fprintf(stderr, "deadstation hands: %.1f at %d %d %d point %d %d forward %d %d %d up %d gear %d revs %d spin %d throttle %d brake %d steering %d clock %u\n", (double)handsSteps / rate,
				(int32_t)guestRead(c + 196), (int32_t)guestRead(c + 200), (int32_t)guestRead(c + 204), (int32_t)guestRead(c + 476), (int32_t)guestRead(c + 484), guestHalf(c + 180), guestHalf(c + 186), guestHalf(c + 192), guestHalf(c + 184),
				(int8_t)guestByte(c + 216), (int32_t)guestRead(c + 4060), (int32_t)guestRead(c + 4068), guestHalf(c + Throttle), guestHalf(c + Brake), guestHalf(c + Steering), guestRead(RaceClock));
	if (rollTrials == 0)
		return;
	const float up = carUpY();
	const uint32_t since = (uint32_t)handsSteps - roll.at;
	const auto to = [&](RollTest::Phase p) { roll.phase = p, roll.at = (uint32_t)handsSteps; };
	switch (roll.phase)
	{
		case RollTest::Waiting:
			if (since > rate * 2)
				to(RollTest::Pushing);
			break;
		case RollTest::Pushing:
			if (up < -0.25f || since > rate * 2)
				to(RollTest::Resting);
			break;
		case RollTest::Resting:
			if (since > rate * 3)
			{
				roll.restUp = guestHalf(c + 184);
				roll.presses = roll.accepted = roll.forced = 0, roll.steps = roll.contactA = roll.contactB = 0;
				roll.startedAt = (uint32_t)handsSteps;
				if (up > 0.7f && wheelsDown())
				{
					std::fprintf(stderr, "deadstation roll: trial %d: it came back by itself, up %.2f\n", roll.trial, up);
					roll.trial++;
					to(roll.trial >= rollTrials ? RollTest::Finished : RollTest::Waiting);
				}
				else
					to(RollTest::Pressing);
			}
			break;
		case RollTest::Pressing:
		{
			roll.steps++;
			const int8_t kick = (int8_t)guestByte(c + Kick);
			const bool a = guestHalf(c + 1916) != 0 && guestHalf(c + 1340) != 0, b = guestHalf(c + 1724) != 0 && guestHalf(c + 1532) != 0;
			roll.contactA += a, roll.contactB += b;
			if (kick == 1 && a)
				roll.forced++;
			if (kick == 2 && b)
				roll.forced++;
			if (kick != 0)
				roll.accepted++;
			const bool done = up > 0.7f && wheelsDown();
			if (done || since > rate * 30)
			{
				std::fprintf(stderr, "deadstation roll: trial %d: rested at up %.2f; %s after %.2f s, %d presses, kick steps %d, of them with force %d; corners down A %.0f%% B %.0f%%; strength reached %d\n", roll.trial, roll.restUp / 4096.0,
						done ? "upright" : "NOT UPRIGHT", (double)since / rate, roll.presses, roll.accepted, roll.forced, 100.0 * roll.contactA / std::max<uint32_t>(roll.steps, 1), 100.0 * roll.contactB / std::max<uint32_t>(roll.steps, 1), guestHalf(c + 4078));
				roll.trial++;
				to(roll.trial >= rollTrials ? RollTest::Finished : RollTest::Waiting);
			}
			break;
		}
		default:
			break;
	}
}


// ---- Medium High: a camera between Medium and Far, higher, looking down the road over the car ----
//
// The game's cameras are three rows of a table by its view (0 in the car, 1 Close, 2 Medium, 3 Far): how far above the car, how far behind it, and how far above it to look. So
// this one is Medium's view with its own numbers in the table while the camera moves. The game's view stays 0 to 3, for all that goes by it; a row more in the pause menu's camera page
// (0x80023B6C, one player) says which. Saved in its own file, as the game keeps no camera between runs of itself.
constexpr uint32_t CameraMove = 0x8002D3F0, CameraMenuBack = 0x80024DD0, ViewChosen = 0x8005649C, ViewNow = 0x80060B60;
constexpr uint32_t CameraAbove = 0x80060BD6 + 4, CameraBehind = 0x80060BDE + 4, CameraLook = 0x80060BE6 + 4;
constexpr int16_t HighAbove = 232, HighBehind = 768, HighLook = 56;
constexpr uint32_t MenuPadPressed = PadStates + 8, MenuDown = 0x4000, MenuUp = 0x1000;
constexpr uint32_t MenuText = 0x80041038, MenuFarBack = 0x80024A94, MenuPanel = 0x8004322C, MenuPanelBack = 0x80024A08;
constexpr int MenuRowHeight = 11, MenuHighlightTop = 97, MenuPanelBottom = 0x8E, MenuPanelRight = 0xC5;
inline uint32_t menuBuffer(CPUState *cpu) { return guestByte(cpu->gpr[18] + 0x20000 - 25448); }
void cameraMenu(CPUState *cpu)
{
	const uint32_t pressed = psx_mod_read_half(MenuPadPressed);
	const int now = guestByte(ViewChosen);
	// (the row before this press: the game has moved its view by one already)
	int view = now;
	if ((pressed & MenuDown) != 0)
		view = (now + 3) % 4;
	else if ((pressed & MenuUp) != 0)
		view = (now + 1) % 4;
	int row = view < 2 ? view : view == 2 ? (mediumHigh ? 3 : 2) : 4;
	if ((pressed & MenuDown) != 0)
		row = (row + 1) % 5;
	else if ((pressed & MenuUp) != 0)
		row = (row + 4) % 5;
	const int chosen = row < 3 ? row : row == 3 ? 2 : 3;
	const bool high = row == 3;
	if (high != mediumHigh)
	{
		mediumHigh = high;
		if (FILE *f = std::fopen(CameraFile, "wb"))
		{
			std::fputc(high ? 1 : 0, f);
			std::fclose(f);
		}
	}
	guestByte(ViewChosen) = (uint8_t)chosen;
	psx_mod_write_half(ViewNow, (uint16_t)chosen);
	cpu->gpr[6] = (uint32_t)chosen;
	// (the highlight on its row, in the bar the page just placed by the game's view)
	const uint32_t bar = cpu->gpr[29] + 24 + menuBuffer(cpu) * 28;
	psx_mod_write_half(bar + 10, (uint16_t)(MenuHighlightTop + row * MenuRowHeight));
}
bool inCameraMove = false;
int onCameraMove(CPUState *cpu, uint32_t)
{
	if (inCameraMove)
		return 0;
	const uint32_t ra = cpu->gpr[31];
	if (ra == CameraMenuBack)
		cameraMenu(cpu);
	if (!mediumHigh || (uint16_t)cpu->gpr[6] != 2 || guestByte(Players) > 1)
		return 0;
	int16_t& above = guestHalf(CameraAbove), &behind = guestHalf(CameraBehind), &look = guestHalf(CameraLook);
	const int16_t aboveWas = above, behindWas = behind, lookWas = look;
	above = HighAbove, behind = HighBehind, look = HighLook;
	inCameraMove = true;
	psx_dispatch_call(cpu, CameraMove, ra);
	inCameraMove = false;
	above = aboveWas, behind = behindWas, look = lookWas;
	cpu->gpr[31] = ra;
	return 1;
}
// The pause menu's main page: eight rows from a table, RESUME, CAMERA, SOUND and its slider, MUSIC and its slider, RESTART, QUIT, and a cursor the game keeps in s4 by those eight. So a ninth is
// here, HUD, after CAMERA: drawn in its place with those below it a row down, and the cursor on it held on CAMERA with a word here, the buttons seen first. Its own page takes the main page's
// place, the game's rows not drawn and no button reaching them but Start.
constexpr uint32_t PausePadsBack = 0x80023DD0, PauseTrackBack = 0x80024E48, MainRowBack = 0x80024074, MainPanelBack = 0x80024044, ConfirmPanelBack = 0x8002465C;
constexpr uint32_t PadHeld = PadStates, PadPressed = PadStates + 8, PadReleased = PadStates + 10, MenuSound = 0x8003FB30;
constexpr uint32_t PadStart = 0x800, PadCross = 0x40, PadCircle = 0x20, PadTriangle = 0x10, PadLeft = 0x8000, PadRight = 0x2000;
constexpr int MainFirstRow = 0x44, MainHighlightTop = 67, MainPanelBottom = 0x9C, MainRows = 8, MainHudRow = 2;
constexpr int HudLeft = 84, HudRight = 236, HudLabelX = 93, HudValueX = 196;
bool onHudRow = false, onHudPage = false;
int hudRow = 0;
inline void saveHud()
{
	if (FILE *f = std::fopen(HudFile, "wb"))
	{
		const uint8_t kept[2] = { (uint8_t)hudOff, (uint8_t)(hudOff >> 8) };
		std::fwrite(kept, 1, 2, f);
		std::fclose(f);
	}
}
inline void menuBeep(CPUState *cpu, uint32_t ra)
{
	cpu->gpr[4] = 0xA, cpu->gpr[5] = 0, cpu->gpr[6] = 0x3FFF, cpu->gpr[7] = 0x3FFF;
	psx_dispatch_call(cpu, MenuSound, ra);
	cpu->gpr[31] = ra;
}
// the pause menu's buttons, as they're read
void pauseButtons(CPUState *cpu, uint32_t ra)
{
	uint32_t pressed = psx_mod_read_half(PadPressed);
	uint32_t &cursor = cpu->gpr[20];		// (s4)
	const uint32_t page = cpu->gpr[23];		// (s7: 0 the main page)
	if (page != 0)
	{
		onHudRow = onHudPage = false;
		return;
	}
	if ((pressed & PadStart) != 0)
		onHudRow = onHudPage = false;
	if (onHudPage)
	{
		bool beep = false;
		if ((pressed & MenuDown) != 0)
			hudRow = (hudRow + 1) % (HudParts + 1), beep = true;
		if ((pressed & MenuUp) != 0)
			hudRow = (hudRow + HudParts) % (HudParts + 1), beep = true;
		if ((pressed & (PadCircle | PadTriangle)) != 0 || ((pressed & PadCross) != 0 && hudRow == HudParts))
			onHudPage = false, beep = true;
		else if ((pressed & (PadCross | PadLeft | PadRight)) != 0 && hudRow < HudParts)
		{
			hudOff ^= 1u << hudRow;
			saveHud();
			beep = true;
		}
		// (nothing but Start for the game)
		psx_mod_write_half(PadPressed, (uint16_t)(pressed & PadStart));
		psx_mod_write_half(PadHeld, (uint16_t)(psx_mod_read_half(PadHeld) & PadStart));
		psx_mod_write_half(PadReleased, 0);
		if (beep)
			menuBeep(cpu, ra);
		return;
	}
	if (onHudRow)
	{
		if ((pressed & MenuDown) != 0)
			onHudRow = false;		// (the game goes on from CAMERA, to SOUND)
		else if ((pressed & MenuUp) != 0)
		{
			onHudRow = false;
			psx_mod_write_half(PadPressed, (uint16_t)(pressed & ~MenuUp));
			menuBeep(cpu, ra);
		}
		else if ((pressed & PadCross) != 0)
		{
			onHudPage = true, hudRow = 0;
			psx_mod_write_half(PadPressed, (uint16_t)(pressed & ~PadCross));
			menuBeep(cpu, ra);
		}
		return;
	}
	if (cursor == 1 && (pressed & MenuDown) != 0)
	{
		onHudRow = true;
		psx_mod_write_half(PadPressed, (uint16_t)(pressed & ~MenuDown));
		menuBeep(cpu, ra);
	}
	else if (cursor == 2 && (pressed & MenuUp) != 0)
	{
		onHudRow = true, cursor = 1;
		psx_mod_write_half(PadPressed, (uint16_t)(pressed & ~MenuUp));
		menuBeep(cpu, ra);
	}
}
// the highlight, where the game placed it by its own rows, before the frame's drawn
void pauseHighlight(CPUState *cpu)
{
	if (cpu->gpr[23] == 2)
		return;		// (the camera's page: cameraMenu)
	const uint32_t bar = cpu->gpr[29] + 24 + menuBuffer(cpu) * 28;
	if (cpu->gpr[23] != 0)
	{
		psx_mod_write_half(bar + 8, 0x7B);
		return;
	}
	const int row = onHudRow ? MainHudRow : (int)cpu->gpr[20] + ((int)cpu->gpr[20] >= MainHudRow ? 1 : 0);
	psx_mod_write_half(bar + 8, onHudPage ? (uint16_t)HudLeft : (uint16_t)0x7B);
	psx_mod_write_half(bar + 10, (uint16_t)(MainHighlightTop + (onHudPage ? hudRow : row) * MenuRowHeight));
}
// the rows: on the main page HUD among them; on the camera page Medium High where Far was, Far a row down; on the HUD's page its own
bool inMenuText = false;
int onMenuText(CPUState *cpu, uint32_t)
{
	const uint32_t ra = cpu->gpr[31];
	if (inMenuText)
		return 0;
	const uint32_t text = cpu->gpr[4], font = cpu->gpr[5], x = cpu->gpr[6], y = cpu->gpr[7];
	const auto say = [&](uint32_t what, uint32_t atX, uint32_t atY) {
		cpu->gpr[4] = what, cpu->gpr[5] = font, cpu->gpr[6] = atX, cpu->gpr[7] = atY;
		psx_dispatch_call(cpu, MenuText, ra);
	};
	if (ra == MenuFarBack && pages.mediumHigh != 0)
	{
		inMenuText = true;
		say(pages.mediumHigh, x, y);
		say(text, x, y + MenuRowHeight);
		inMenuText = false;
		cpu->gpr[31] = ra;
		return 1;
	}
	if (ra != MainRowBack || hudTextHud == 0)
		return 0;
	const int row = ((int)y - MainFirstRow) / MenuRowHeight;
	inMenuText = true;
	if (onHudPage)
	{
		if (row == 0)
			for (int i = 0; i <= HudParts; i++)
			{
				const uint32_t atY = (uint32_t)(MainFirstRow + i * MenuRowHeight);
				if (i == HudParts)
					say(hudTextDone, HudLabelX, atY);
				else
				{
					say(hudTexts[i], HudLabelX, atY);
					say((hudOff >> i & 1) != 0 ? hudTextOff : hudTextOn, HudValueX, atY);
				}
			}
	}
	else
	{
		if (row == MainHudRow)
			say(hudTextHud, x, (uint32_t)(MainFirstRow + MainHudRow * MenuRowHeight));
		say(text, x, y + (row >= MainHudRow ? MenuRowHeight : 0));
	}
	inMenuText = false;
	cpu->gpr[31] = ra;
	return 1;
}
// and the panel behind them as long and wide as they want; a quad the game places, kept for its other pages as it was
void onMenuPanel(CPUState *cpu, uint32_t)
{
	const uint32_t ra = cpu->gpr[31];
	const bool cameraPage = ra == MenuPanelBack, mainPage = ra == MainPanelBack;
	if (!cameraPage && !mainPage && ra != ConfirmPanelBack)
		return;
	const uint32_t panel = cpu->gpr[29] + 80 + menuBuffer(cpu) * 24;
	const uint16_t left = (uint16_t)(mainPage && onHudPage ? HudLeft : 0x7B);
	const uint16_t right = (uint16_t)(cameraPage ? MenuPanelRight + 24 : mainPage && onHudPage ? HudRight : MenuPanelRight);
	psx_mod_write_half(panel + 8, left), psx_mod_write_half(panel + 16, left);
	psx_mod_write_half(panel + 12, right), psx_mod_write_half(panel + 20, right);
	if (cameraPage || mainPage)
	{
		const int bottom = cameraPage ? MenuPanelBottom + MenuRowHeight : onHudPage ? MainFirstRow + (HudParts + 1) * MenuRowHeight + 3 : MainPanelBottom + MenuRowHeight;
		psx_mod_write_half(panel + 18, (uint16_t)bottom);
		psx_mod_write_half(panel + 22, (uint16_t)bottom);
	}
}

// ---- Out of a race: a cut to black, then the menu ----
//
// What the game did on its way back (its next screen going into VRAM as a frame still on show, in pieces) is not shown. The frames are painted black, and nothing is presented until the game draws
// a frame of its next screen. (A loading that never draws one, past 30 seconds: shown again, and said.)
constexpr uint32_t CdStop = 0x800449C4, RaceOverBackA = 0x80020F40, RaceOverBackB = 0x8002208C, PauseOpenBack = 0x80023C0C;
bool leavingRace = false;
uint32_t leavingBlanks = 0;
void onCdStop(CPUState *cpu, uint32_t)
{
	const uint32_t ra = cpu->gpr[31];
	if (ra == PauseOpenBack)
		onHudRow = onHudPage = false;
	if ((ra != RaceOverBackA && ra != RaceOverBackB) || blackFill == 0)
		return;
	const uint32_t at = blackFill & 0x1FFFFF;
	guestWord(at, 3u << 24 | ((blackFill + 16) & 0xFFFFFF));
	guestWord(at + 4, 0x02000000), guestWord(at + 8, 0), guestWord(at + 12, 240u << 16 | 512u);
	guestWord(at + 16, 3u << 24 | 0xFFFFFF);
	guestWord(at + 20, 0x02000000), guestWord(at + 24, 512u), guestWord(at + 28, 240u << 16 | 512u);
	deadstation_scene_perspective(0);
	psx_native_gpu_list(at);
	leavingRace = true, leavingBlanks = 0;
}

// The camera, on a string behind the car: its rise when something's in the way of it, or fall when not (0x8002DCD4, what's in the way as a fraction,
// and then 0x8002D8D4's), and its swing round behind the car (0x8004BB7C, from 0x8002DB30), a pass at half
constexpr uint32_t CameraWall = 0x8002DCD4, CameraWallBack = 0x8002D8C8, CameraSwing = 0x8004BB7C, CameraSwingBack = 0x8002DB38;
bool inCameraWall = false;
int onCameraWall(CPUState *cpu, uint32_t)
{
	const uint32_t ra = cpu->gpr[31], camera = cpu->gpr[4];
	if (inCameraWall || !sixtyNow(ra, CameraWallBack))
		return 0;
	inCameraWall = true;
	psx_dispatch_call(cpu, CameraWall, ra);
	inCameraWall = false;
	cpu->gpr[31] = ra;
	const int32_t wall = (int32_t)cpu->gpr[2];
	if (wall != 0)
		cpu->gpr[2] = (uint32_t)(wall / 2 != 0 ? wall / 2 : (wall > 0 ? 1 : -1));
	else
		guestWord(camera + 68, guestRead(camera + 68) - 2);	// (it adds 4)
	return 1;
}
void onCameraSwing(CPUState *cpu, uint32_t)
{
	if (cpu->gpr[31] != CameraSwingBack || !sixty || guestByte(Players) > 1)
		return;
	int16_t& angle = guestHalf(cpu->gpr[4] + 2);
	angle = (int16_t)cameraSwing.move(angle);
}

// What cars throw up, 0x800373C0: the whole of it on the passes it counts on. On those between, if there's a frame, the objects it keeps are put in the tiles'
// lists (they're emptied each pass, 0x80038A20), as its end does, those flying half a step on
constexpr uint32_t WorldEffects = 0x800373C0, WorldEffectsBack = 0x800206F0, TileIndexOf = 0x80060D74, TileListed = Gp + 904;
constexpr uint32_t Grit = 0x80065E14;			// 32 of 72 bytes: +2 in use, +4,+6,+8 its place, whole parts of +52,+56,+60; +36,+40,+44 a step's move
constexpr uint32_t Smoke = 0x80064B40;			// 8 x 8 of 44 bytes: +2 in use, +16,+18 its size
constexpr uint32_t Marks = 0x800612F8;			// 8 x 15 of 112 bytes: +2 in use, +46 finished
constexpr uint32_t Spray = 0x80066A2C;			// 8 x 4 of 52 bytes: +2 in use, +16,+18 its size
constexpr uint32_t Flakes = 0x800B749C, FlakesOn = 0x80060B50;	// 64 of 40 bytes: +2 in use, +6 height; +36 a step's fall (a track's own: leaves...)
void listObject(uint32_t object)
{
	if (guestByte(object) >= 11)
		return;
	const int x = (guestHalf(object + 4) >> 9) + 32, z = (guestHalf(object + 8) >> 9) + 32;
	const uint32_t index = guestRead(TileIndexOf);
	const uint16_t tile = (uint16_t)guestHalf(index + (uint32_t)(z << 7) + (uint32_t)(x << 1));
	if (tile == 0xFFFF)
	{
		guestByte(object + 3) = 1;
		return;
	}
	const uint32_t at = index + 8364 + (uint32_t)tile * 88, first = guestRead(at + 68);
	if (first == 0 || guestByte(first + 3) == 0xFF)
	{
		guestWord(at + 72, guestRead(TileListed));
		guestWord(TileListed, at);
	}
	guestWord(object + 12, first);
	guestWord(at + 68, object);
}
struct SmokeSize { int16_t w, h, lastW, lastH; bool grown; };
SmokeSize smokeSizes[8 * 8 + 8 * 4];
int16_t flakesAhead[64];
inline uint32_t smokeAt(int i) { return i < 64 ? Smoke + (uint32_t)i * 44 : Spray + (uint32_t)(i - 64) * 52; }
int onWorldEffects(CPUState *cpu, uint32_t)
{
	if (!sixtyNow(cpu->gpr[31], WorldEffectsBack))
		return 0;
	if (tickPass)
	{
		// (back to what the game left, before it goes on from there)
		for (int i = 0; i < 96; i++)
		{
			SmokeSize& s = smokeSizes[i];
			if (s.grown)
			{
				const uint32_t o = smokeAt(i);
				if (guestByte(o + 2) != 0)
					guestHalf(o + 16) = s.w, guestHalf(o + 18) = s.h;
				s.grown = false;
			}
		}
		for (int i = 0; i < 64; i++)
			if (flakesAhead[i] != 0)
			{
				guestHalf(Flakes + (uint32_t)i * 40 + 6) -= flakesAhead[i];
				flakesAhead[i] = 0;
			}
		return 0;
	}
	if (guestRead(DrawingFlag) == 0)
		return 1;
	if (guestByte(FlakesOn) != 0)
		for (int i = 0; i < 64; i++)
		{
			const uint32_t o = Flakes + (uint32_t)i * 40;
			if (guestByte(o + 2) != 1)
				continue;
			flakesAhead[i] = (int16_t)(guestHalf(o + 36) / 2);
			guestHalf(o + 6) += flakesAhead[i];
			listObject(o);
		}
	for (int i = 0; i < 32; i++)
	{
		const uint32_t o = Grit + (uint32_t)i * 72;
		if (guestByte(o + 2) == 0)
			continue;
		guestHalf(o + 4) = (int16_t)(((int32_t)guestRead(o + 52) + (int32_t)guestRead(o + 36) / 2) >> 16);
		guestHalf(o + 6) = (int16_t)(((int32_t)guestRead(o + 56) + (int32_t)guestRead(o + 40) / 2) >> 16);
		guestHalf(o + 8) = (int16_t)(((int32_t)guestRead(o + 60) + (int32_t)guestRead(o + 44) / 2) >> 16);
		listObject(o);
	}
	for (int i = 0; i < 96; i++)
	{
		const uint32_t o = smokeAt(i);
		if (guestByte(o + 2) == 0)
			continue;
		SmokeSize& s = smokeSizes[i];
		const int16_t w = guestHalf(o + 16), h = guestHalf(o + 18);
		s.w = w, s.h = h;
		// (half what it grew by, the last pass it did; a new one hasn't grown yet)
		const int grewW = w - s.lastW, grewH = h - s.lastH;
		guestHalf(o + 16) = (int16_t)(w + (std::abs(grewW) < 1024 ? grewW / 2 : 0)), guestHalf(o + 18) = (int16_t)(h + (std::abs(grewH) < 1024 ? grewH / 2 : 0));
		s.lastW = w, s.lastH = h;
		s.grown = true;
		listObject(o);
	}
	for (int i = 0; i < 8 * 15; i++)
	{
		const uint32_t o = Marks + (uint32_t)i * 112;
		if (guestByte(o + 2) != 0 && guestByte(o + 46) != 0)
			listObject(o);
	}
	return 1;
}

// The wait for a blank is no work at all: the guest's clock goes straight to it, where the pacer holds real time (the game's poll, at a faster guest CPU, took a
// third of the race's time on this thread, in nothing but counting up to it)
void jumpToBlank()
{
	const uint32_t rate = psx_cycle_rate;
	psx_cycle_rate = 256u;
	psx_advance_cycles(interrupts_cycles_to_vblank() + 1u);
	psx_cycle_rate = rate;
}
// The frame's wait for two blanks (0x8002090x, on 0x80058D78): for one. The game wants 2 there, then starts again from 0
constexpr uint32_t DrawDone = 0x800483F8, DrawDoneBack = 0x800208E8;
bool inDrawDone = false;
int onDrawDone(CPUState *cpu, uint32_t)
{
	const uint32_t ra = cpu->gpr[31];
	if (inDrawDone || !sixtyNow(ra, DrawDoneBack))
		return 0;
	inDrawDone = true;
	psx_dispatch_call(cpu, DrawDone, ra);
	inDrawDone = false;
	cpu->gpr[31] = ra;
	workEnded = Clock::now();
	if (guestRead(BlankCount) >= 1)
		guestWord(BlankCount, 2);
	else
	{
		guestWord(BlankCount, 1);
		// (its interrupt, on the way out of here, counts it)
		jumpToBlank();
	}
	return 1;
}
// the wait's over, the blank's been (0x80046034, the loop's next call): when the time's read, before this frame's list goes out
constexpr uint32_t AfterTheWait = 0x80046034, AfterTheWaitBack = 0x80020908;
Clock::time_point blankAt {};
void onAfterTheWait(CPUState *cpu, uint32_t)
{
	if (cpu->gpr[31] == AfterTheWaitBack && sixty)
		blankAt = Clock::now();
}
// a pass with nothing to draw waits for its blank here (VSync(0), what it returns not looked at)
constexpr uint32_t VSync = 0x8004663C, VSyncBack = 0x80020968;
// The menus at 60. Their loop (0x80017B30) waits two blanks a frame, VSync(2): here one. What's counted in frames there keeps real time, on every second one: its counter (what blinks,
// gp+1564) and what's counted on the pads (held, and how long since one was touched, for the demo).
constexpr uint32_t MenuVSyncBack = 0x80017C6C, MenuPadsBack = 0x80017BA0, MenuFrames = Gp + 1564;
bool menusAtSixty = true, menuTick = true;
int onVSync(CPUState *cpu, uint32_t)
{
	if (menusAtSixty && cpu->gpr[31] == MenuVSyncBack)
	{
		cpu->gpr[4] = 0;
		return 0;
	}
	if (cpu->gpr[31] != VSyncBack || !sixtyRunning)
		return 0;
	workEnded = Clock::now();
	// (no frame, so nothing to show: the time passes all the same, for the sound and the rest, and the pacer lets it, being owed)
	deadstation_no_present = 1;
	jumpToBlank();
	deadstation_no_present = 0;
	return 1;
}
// A pass ends: the race's bookkeeping, every pass, its clock on every second. How the time stands says if the next pass draws
int onBookkeeping(CPUState *cpu, uint32_t)
{
	if (!sixtyNow(cpu->gpr[31], BookkeepingBack))
		return 0;
	if (!tickPass)
	{
		// (it adds one to each racing car's two clocks, and may use them on the way: taken off first)
		const int cars = std::min<int>(guestByte(CarCount), 8);
		for (int i = 0; i < cars; i++)
		{
			const uint32_t o = CarObjects + (uint32_t)i * CarObjectSize;
			if ((int8_t)guestByte(o + 63) == 1)
			{
				guestWord(o + 68, guestRead(o + 68) - 1);
				guestWord(o + 72, guestRead(o + 72) - 1);
			}
		}
	}
	tickPass = !tickPass;
	const bool drew = guestRead(DrawingFlag) != 0;
	if (!sixtyRunning)
	{
		guestWord(DrawingFlag, 1);
		keptFrom = Clock::time_point {};
		return 0;
	}
	if (!drew)
		guestWord(BlankCount, 0);	// (that pass waited without it)
	const auto now = Clock::now();
	const double period = deadstation_frame_period_ms;
	if (stats && passStarted != Clock::time_point {})
	{
		const double ms = std::chrono::duration<double, std::milli>(workEnded - passStarted).count();
		if (drew)
		{
			passTimes.drawn += ms, passTimes.drawnCount++;
			passTimes.list += std::chrono::duration<double, std::milli>(now - blankAt).count();
		}
		else
			passTimes.undrawn += ms, passTimes.undrawnCount++;
	}
	// Behind? Read at the blank each pass waited for: the same point in every kind of pass (after it, a drawn one still has its list to send). The pacer holds a pass at its
	// blank only when it's on time, and lets them run on while it's owed, and so does this: ahead, the time's as it should be from here; and a long stop isn't owed, it
	// forgives that too.
	const auto at = drew ? blankAt : now;
	const double kept = std::chrono::duration<double, std::milli>(at - keptFrom).count();
	// (a step with nothing to present has nothing to wait for but the pacer: if it took as long as a frame, the pacer held it, and it's owed nothing, whatever was counted here. The
	// count can drift from the pacer's, by a blank in the middle of a pass, and steps the pacer holds would never win it back)
	const bool held = !drew && std::chrono::duration<double, std::milli>(now - passStarted).count() > period * 0.9;
	if (keptFrom == Clock::time_point {} || held || kept > (double)passesKept * period + period * 12)
	{
		keptFrom = at;
		passesKept = 0;
	}
	else
	{
		passesKept++;
		const double behind = kept - (double)passesKept * period;
		if (behind < 0)
		{
			keptFrom = at;
			passesKept = 0;
		}
		else if (behind >= period && passesUndrawn < MostUndrawn)
		{
			passesUndrawn++;
			guestWord(DrawingFlag, 0);
			return 0;
		}
	}
	passesUndrawn = 0;
	guestWord(DrawingFlag, 1);
	return 0;
}
// (where everyone is, on the game's own clock: with the AI driving, two runs are the same race if these are the same)
inline void logPlaces()
{
	static uint32_t placeAt = 0;
	const uint32_t clock = guestRead(0x800659C0);
	if (stats && clock != placeAt && clock % 30 == 0)
		std::fprintf(stderr, "deadstation rallycross: at clock %u: %d %d, %d %d, %d %d, %d %d; laps %u %u %u %u\n", clock, ramHalf(CarObjects + 4), ramHalf(CarObjects + 8), ramHalf(CarObjects + CarObjectSize + 4), ramHalf(CarObjects + CarObjectSize + 8),
				ramHalf(CarObjects + CarObjectSize * 2 + 4), ramHalf(CarObjects + CarObjectSize * 2 + 8), ramHalf(CarObjects + CarObjectSize * 3 + 4), ramHalf(CarObjects + CarObjectSize * 3 + 8),
				guestByte(CarObjects + 62), guestByte(CarObjects + CarObjectSize + 62), guestByte(CarObjects + CarObjectSize * 2 + 62), guestByte(CarObjects + CarObjectSize * 3 + 62));
	if (stats && clock != placeAt && clock % 30 == 0)
	{
		// (the first car's drive: gear, revs as its gearbox sees them, what its wheels are turning at, throttle, brake)
		const uint32_t car = 0x800B3448;
		std::fprintf(stderr, "deadstation rallycross: drive at %u: gear %d revs %d spin %d throttle %d brake %d surfaces %u %u %u %u; contact %d %d %d %d\n", clock, (int8_t)guestByte(car + 216), (int32_t)guestRead(car + 4060), (int32_t)guestRead(car + 4068), ramHalf(car + 210), ramHalf(car + 212),
				guestByte(car + 631), guestByte(car + 631 + 192), guestByte(car + 631 + 384), guestByte(car + 631 + 576), ramHalf(car + 572), ramHalf(car + 572 + 192), ramHalf(car + 572 + 384), ramHalf(car + 572 + 576));
	}
	placeAt = clock;
}

// the pads, each pass: what the game counts on them (held, how long a pad's been gone, for the demo's end and the pause) on every second
int onPadsRead(CPUState *cpu, uint32_t)
{
	const uint32_t ra = cpu->gpr[31];
	if (!inPads && ra == PausePadsBack)
	{
		inPads = true;
		psx_dispatch_call(cpu, PadsRead, ra);
		inPads = false;
		cpu->gpr[31] = ra;
		pauseButtons(cpu, ra);
		return 1;
	}
	if (!inPads && menusAtSixty && ra == MenuPadsBack)
	{
		uint32_t counts[4][2];
		for (uint32_t i = 0; i < 4; i++)
			counts[i][0] = guestRead(PadStates + i * 24 + 16), counts[i][1] = guestRead(PadStates + i * 24 + 20);
		inPads = true;
		psx_dispatch_call(cpu, PadsRead, ra);
		inPads = false;
		cpu->gpr[31] = ra;
		menuTick = !menuTick;
		if (!menuTick)
		{
			guestWord(MenuFrames, guestRead(MenuFrames) - 1);
			for (uint32_t i = 0; i < 4; i++)
				for (uint32_t j = 0; j < 2; j++)
					if (guestRead(PadStates + i * 24 + 16 + j * 4) == counts[i][j] + 1)
						guestWord(PadStates + i * 24 + 16 + j * 4, counts[i][j]);
		}
		return 1;
	}
	if (inPads || !sixtyNow(ra, PadsBack))
		return 0;
	// (the replay camera's orbit, by the buttons, 16 a pass: it's run on after last pass's bookkeeping)
	static uint32_t orbitWas = 0;
	if (guestByte(ReplayMode) == 1)
	{
		const uint32_t orbit = guestRead(ReplayOrbit);
		const int delta = (int)(((orbit - orbitWas + 0x800) & 0xFFF)) - 0x800;
		orbitWas = (orbitWas + (uint32_t)(delta / 2)) & 0xFFF;
		guestWord(ReplayOrbit, orbitWas);
	}
	passStarted = Clock::now();
	uint32_t counts[4][2];
	for (uint32_t i = 0; i < 4; i++)
		counts[i][0] = guestRead(PadStates + i * 24 + 16), counts[i][1] = guestRead(PadStates + i * 24 + 20);
	inPads = true;
	psx_dispatch_call(cpu, PadsRead, ra);
	inPads = false;
	cpu->gpr[31] = ra;
	if (!tickPass)
		for (uint32_t i = 0; i < 4; i++)
			for (uint32_t j = 0; j < 2; j++)
				if (guestRead(PadStates + i * 24 + 16 + j * 4) == counts[i][j] + 1)
					guestWord(PadStates + i * 24 + 16 + j * 4, counts[i][j]);
	return 1;
}

// the car renderer's limit (its depth / 4 - 24 past the ordering table's 1536: no car): what its depth call says, held short of that. A car
// that far has its polygons' own depths; it only wants not to be dropped.
constexpr uint32_t CarDepthCall = 0x8004B9EC, CarDepthReturn = 0x8001339C, MarkDepthReturn = 0x80012ABC;	// (the same for the skid marks, 0x800129A0)
constexpr int32_t CarDepthMost = (1535 + 24) * 4, MarkDepthMost = (1535 + 7) * 4;
bool inCarDepth = false;
int onCarDepth(CPUState *cpu, uint32_t)
{
	const uint32_t ra = cpu->gpr[31];
	if (inCarDepth || (ra != CarDepthReturn && ra != MarkDepthReturn))
		return 0;
	inCarDepth = true;
	psx_dispatch_call(cpu, CarDepthCall, ra);
	inCarDepth = false;
	cpu->gpr[31] = ra;
	const int32_t most = ra == CarDepthReturn ? CarDepthMost : MarkDepthMost;
	if ((int32_t)cpu->gpr[2] > most)
		cpu->gpr[2] = (uint32_t)most;
	return 1;
}

// DEADSTATION_RALLYCROSS_STATS, each vertical blank: how far each car has gone and the race's clock
void onVBlank()
{
	logPlaces();
	blanksSeen++;
	if (leavingRace)
	{
		// (the black takes a blank or two to be on show; from there nothing else is)
		if (++leavingBlanks > 2)
			deadstation_no_present = 1;
		if (leavingBlanks > 1800)
		{
			std::fprintf(stderr, "deadstation rallycross: 30 seconds out of a race with no next screen drawn: showing what there is\n");
			leavingRace = false;
		}
	}
	if (!stats)
		return;
	static float lastX[8], lastZ[8], travelled[8];
	static unsigned frames = 0;
	const int cars = std::min<int>(guestByte(CarCount), 8);
	for (int i = 0; i < cars; i++)
	{
		const uint32_t record = guestRead(CarObjects + (uint32_t)i * CarObjectSize + CarRecordAt);
		if ((record >> 24) != 0x80)
			continue;
		const float x = (float)(int32_t)guestRead(record + 196), z = (float)(int32_t)guestRead(record + 204);
		const float d = std::sqrt((x - lastX[i]) * (x - lastX[i]) + (z - lastZ[i]) * (z - lastZ[i]));
		if (d < 2000)	// (not the jump back to the start)
			travelled[i] += d;
		lastX[i] = x, lastZ[i] = z;
	}
	if (++frames % 300 == 0)
	{
		static Clock::time_point printed = Clock::now();
		const auto nowClock = Clock::now();
		const double realSeconds = std::chrono::duration<double>(nowClock - printed).count();
		printed = nowClock;
		static uint32_t clockWas = 0;
		const uint32_t clockNow = guestRead(0x800659C0);
		std::fprintf(stderr, "deadstation rallycross: %u blanks; clock %u; state %u; travelled %.0f %.0f %.0f %.0f %.0f %.0f; %u cars, %u shadow polygons\n", frames, clockNow, psx_mod_read_half(0x80060B6A),
				travelled[0], travelled[1], travelled[2], travelled[3], travelled[4], travelled[5], statCars, statPolygons);
		// (the game's speed, by its own clock against real time: it counts 30 a second)
		std::fprintf(stderr, "deadstation rallycross: game speed %.1f%% (%u clock ticks in %.2f s); %d cars in the race\n", clockNow >= clockWas ? (double)(clockNow - clockWas) / 30.0 / realSeconds * 100.0 : 0.0,
				clockNow - clockWas, realSeconds, (int)guestByte(0x8005649A));
		// (each car's laps, and the time of each in the game's own ticks)
		for (uint32_t i = 0; i < 4; i++)
		{
			const uint32_t o = CarObjects + i * CarObjectSize;
			std::fprintf(stderr, "deadstation rallycross: car %u: %u laps; times %u %u %u %u %u\n", i, guestByte(o + 62), guestRead(o + 76), guestRead(o + 80), guestRead(o + 84), guestRead(o + 88), guestRead(o + 92));
		}
		clockWas = clockNow;
		const unsigned d = std::max(passTimes.drawnCount, 1u), u = std::max(passTimes.undrawnCount, 1u), all = std::max(passTimes.drawnCount + passTimes.undrawnCount, 1u);
		std::fprintf(stderr, "deadstation rallycross: passes %u drawn at %.1f ms, %u not at %.1f ms; a pass: brains %.1f, physics %.1f; a drawn one: track %.1f, cars %.1f, then sending its list %.1f; %zu solid + %zu blended triangles, %zu rain\n", passTimes.drawnCount, passTimes.drawn / d,
				passTimes.undrawnCount, passTimes.undrawn / u, passTimes.brains / all, physicsMs / all, passTimes.track / d, passTimes.cars / d, passTimes.list / d, trackTriangles.size(), trackBlended.size(), fallingNow.size());
		passTimes = PassTimes {};
		physicsMs = 0;
	}
}

// DEADSTATION_RALLYCROSS_COST with _FREEZE: the camera held and each thing taken away in turn, for what each costs the GPU, by the log's "GPU n ms" in one run on one view
bool costing = false;
int costStep = 0;
constexpr const char *CostNames[] = { "all of it", "no rain or snow", "no cones", "no beams", "no shadows", "no blended quads or glass", "no cars", "no solid quads or marks",
		"no cut-outs in the mesh", "no mesh" };
void costNext()
{
	if (costStep + 1 >= (int)(sizeof(CostNames) / sizeof(CostNames[0])))
		return;
	costStep++;
	std::fprintf(stderr, "deadstation rallycross: cost step %d: %s\n", costStep, CostNames[costStep]);
	deadstation_world_skip = costStep >= 9 ? 63 : costStep >= 8 ? 10 : 0;
}

// A race's frame, whole, in the order a frame wants: none of it through the ordering table but the HUD, which the game keeps at the table's entry 0
bool frameDrawn = false;
void emitFrame()
{
	int32_t ox = 0, oy = 0;
	psx_gpu_draw_offset(&ox, &oy);
	trackCamera.drawX = (float)ox, trackCamera.drawY = (float)oy;
	const float fogMost = look.fogFar > 0 && look.fogVeils > 0 ? 1.f - std::pow(0.85f, (float)look.fogVeils) : 0.f;
	// (some of the skies of the conditions have nothing above their far hills, for the fog to be seen there)
	if (fogMost > 0)
		deadstation_scene_whole(look.fog.r, look.fog.g, look.fog.b);
	else
		deadstation_scene_whole(0, 0, 0);
	deadstation_replay_object(0);
	currentObject = 0;
	deadstation_fog(look.fogNear, look.fogFar, fogMost, look.fog.r, look.fog.g, look.fog.b);
	if (tintWanted)
		setTint(true);
	prepareWorldBeams();
	worldBeams();
	// What's solid, then the sky where nothing is, then what's cut out, with the sky to show through its holes
	deadstation_world_draw(&trackCamera, trackVisible.data(), trackVisible.size(), 1);
	setTint(false);
	deadstation_world_sky(&trackCamera);
	addStars();
	if (tintWanted)
		setTint(true);
	deadstation_world_draw(&trackCamera, trackVisible.data(), trackVisible.size(), 2);
	if (costStep < 7)
	{
		deadstation_world_quads(&trackCamera, trackQuads.data(), trackQuads.size());
		deadstation_world_quads(&trackCamera, trackMarks.data(), trackMarks.size());
	}
	emitCarsAndLight();
	if (tintWanted)
		setTint(true);
	addLampLight();
	// what's added or blended, farthest first
	if (!trackQuadsBlended.empty() && costStep < 5 && !without("dust"))
	{
		static std::vector<uint32_t> order;
		static std::vector<DeadstationWorldQuad> sorted;
		order.resize(trackQuadsBlended.size());
		for (uint32_t i = 0; i < order.size(); i++)
			order[i] = i;
		std::sort(order.begin(), order.end(), [](uint32_t a, uint32_t b) { return trackQuadsBlendedZ[a] > trackQuadsBlendedZ[b]; });
		sorted.resize(order.size());
		for (size_t i = 0; i < order.size(); i++)
			sorted[i] = trackQuadsBlended[order[i]];
		deadstation_world_quads(&trackCamera, sorted.data(), sorted.size());
	}
	setTint(false);
	emitEffects();
	deadstation_fog(0, 0, 0, 0, 0, 0);
	deadstation_world_numbers(nullptr, nullptr, 0);
	trackActive = false;
	frameDrawn = true;
}
// The HUD natively. The game's HUD code (800412B4, 8004322C) keeps its layout and fills in its records, as it does; all of it goes in one chain, its few kinds of record read here as data, and drawn
// directly: no ordering table walked, nothing through the GPU's list. Anything else in the chain (a game state that does something new): the list, for that frame.
std::vector<DeadstationTriangle> hudPolygons;
std::vector<DeadstationGlow> hudLines;
bool hudLog = false;
// A frame that has to go by the list (the pause menu's panel): the parts turned off taken out of it for that frame, put back as the next is told. (The game's records outlive a frame, some of them.)
std::vector<std::pair<uint32_t, uint32_t>> hudTakenOut;
void hudListRestore()
{
	for (const auto& taken : hudTakenOut)
		if (guestRead(0x80000000u | taken.first) == 0)
			guestWord(0x80000000u | taken.first, taken.second);
	hudTakenOut.clear();
}
void hudListHide(uint32_t from)
{
	if (hudOff == 0)
		return;
	int32_t ox = 0, oy = 0;
	psx_gpu_draw_offset(&ox, &oy);
	const auto coordinate = [](int v) { return (int)(int16_t)((uint32_t)v << 5) >> 5; };
	uint32_t at = from & 0x1FFFFF;
	for (int n = 0; n < 4000 && at != 0x1FFFFF; n++)
	{
		const uint32_t tag = ramWord(at), len = tag >> 24;
		for (uint32_t w = 0; w < len;)
		{
			const uint32_t base = at + 4 + w * 4, word = ramWord(base), code = word >> 24;
			uint32_t size = 0;
			int mx = 0, my = 0;
			if (code >= 0x64 && code < 0x68)
			{
				const uint32_t place = ramWord(base + 4), wh = ramWord(base + 12);
				size = 4, mx = coordinate((int)(place & 0x7FF)) + (int)(wh & 0xFFFF) / 2, my = coordinate((int)((place >> 16) & 0x7FF)) + (int)(wh >> 16) / 2;
			}
			else if (code >= 0x2C && code < 0x30)
			{
				size = 9;
				for (uint32_t j = 0; j < 4; j++)
				{
					const uint32_t place = ramWord(base + 4 + j * 8);
					mx += coordinate((int)(place & 0x7FF)), my += coordinate((int)((place >> 16) & 0x7FF));
				}
				mx /= 4, my /= 4;
			}
			else
			{
				// (only what the HUD is made of is looked at; a length known, past it)
				const uint32_t known = code >= 0x28 && code < 0x2C ? 5 : code >= 0x60 && code < 0x64 ? 3 : code >= 0x40 && code < 0x44 ? 3 : (code >= 0xE1 && code <= 0xE6) || code == 0 ? 1 : 0;
				if (known == 0)
					break;
				w += known;
				continue;
			}
			if (hudHidden(mx - ox, my - oy))
				for (uint32_t j = 0; j < size; j++)
				{
					hudTakenOut.emplace_back(base + j * 4, ramWord(base + j * 4));
					guestWord(0x80000000u | (base + j * 4), 0);
				}
			w += size;
		}
		at = tag & 0x1FFFFF;
	}
}
bool emitHud(uint32_t from)
{
	static const bool off = std::getenv("DEADSTATION_RALLYCROSS_HUDLIST") != nullptr;
	if (off)
		return false;
	int32_t ox = 0, oy = 0;
	// its area and offset first: the whole screen's, if there's a packet saying so at the start (a split screen's)
	uint32_t at = from & 0x1FFFFF;
	{
		const uint32_t tag = ramWord(at), len = tag >> 24;
		if (len != 0 && (ramWord(at + 4) >> 24) == 0xE3)
		{
			guestWord(at, (tag & 0xFF000000u) | 0xFFFFFFu);
			psx_native_gpu_list(at);
			guestWord(at, tag);
			at = tag & 0x1FFFFF;
		}
	}
	psx_gpu_draw_offset(&ox, &oy);
	hudPolygons.clear();
	hudLines.clear();
	uint32_t page = 0;
	const auto coordinate = [](int v) { return (int)(int16_t)((uint32_t)v << 5) >> 5; };	// (11 bits, signed)
	const auto corner = [&](uint32_t word, float& x, float& y) {
		x = (float)coordinate((int)(word & 0x7FF)) + (float)ox, y = (float)coordinate((int)((word >> 16) & 0x7FF)) + (float)oy;
	};
	for (int n = 0; n < 4000 && at != 0x1FFFFF; n++)
	{
		const uint32_t tag = ramWord(at), len = tag >> 24;
		for (uint32_t w = 0; w < len;)
		{
			const uint32_t base = at + 4 + w * 4, word = ramWord(base), code = word >> 24;
			if (code == 0)
				w += 1;
			else if (code == 0xE1)
				page = word & 0x1FF, w += 1;
			else if (code >= 0x64 && code < 0x68)
			{
				// a sprite in the page set, its size given
				float x, y;
				corner(ramWord(base + 4), x, y);
				const uint32_t texture = ramWord(base + 8), size = ramWord(base + 12);
				const float wide = (float)(size & 0xFFFF), tall = (float)(size >> 16);
				if (hudLog)
					std::fprintf(stderr, "deadstation hud: sprite %.0f %.0f size %.0f %.0f\n", x - (float)ox, y - (float)oy, wide, tall);
				if (hudHidden((int)(x - (float)ox + wide / 2), (int)(y - (float)oy + tall / 2)))
				{
					w += 4;
					continue;
				}
				const int u = (int)(texture & 255), v = (int)((texture >> 8) & 255);
				const uint16_t u1 = (uint16_t)std::min(u + (int)wide, 255), v1 = (uint16_t)std::min(v + (int)tall, 255);
				DeadstationTriangle t {};
				t.page = (uint16_t)page, t.clut = (uint16_t)(texture >> 16), t.raw = (code & 1) != 0, t.semi = (code & 2) != 0 ? (uint8_t)(1 + ((page >> 5) & 3)) : 0;
				t.rgb[0] = t.rgb[1] = t.rgb[2] = word & 0xFFFFFF;
				const float xs[4] = { x, x + wide, x, x + wide }, ys[4] = { y, y, y + tall, y + tall };
				const uint16_t uvs[4] = { (uint16_t)(u | v << 8), (uint16_t)(u1 | v << 8), (uint16_t)(u | v1 << 8), (uint16_t)(u1 | v1 << 8) };
				static constexpr int Order[6] = { 0, 1, 2, 1, 3, 2 };
				for (int k = 0; k < 2; k++)
				{
					for (int j = 0; j < 3; j++)
						t.x[j] = xs[Order[k * 3 + j]], t.y[j] = ys[Order[k * 3 + j]], t.uv[j] = uvs[Order[k * 3 + j]];
					hudPolygons.push_back(t);
				}
				w += 4;
			}
			else if (code >= 0x2C && code < 0x30)
			{
				// a textured quad, the page its own
				float xs[4], ys[4];
				uint16_t uvs[4];
				static constexpr uint32_t Place[4] = { 4, 12, 20, 28 }, Texture[4] = { 8, 16, 24, 32 };
				for (int j = 0; j < 4; j++)
				{
					corner(ramWord(base + Place[j]), xs[j], ys[j]);
					uvs[j] = (uint16_t)ramWord(base + Texture[j]);
				}
				const float middleX = (xs[0] + xs[1] + xs[2] + xs[3]) / 4 - (float)ox, middleY = (ys[0] + ys[1] + ys[2] + ys[3]) / 4 - (float)oy;
				if (hudLog)
					std::fprintf(stderr, "deadstation hud: quad at %.0f %.0f\n", middleX, middleY);
				if (hudHidden((int)middleX, (int)middleY))
				{
					w += 9;
					continue;
				}
				const uint16_t own = (uint16_t)((ramWord(base + 16) >> 16) & 0x1FF);
				DeadstationTriangle t {};
				t.page = own, t.clut = (uint16_t)(ramWord(base + 8) >> 16), t.raw = (code & 1) != 0, t.semi = (code & 2) != 0 ? (uint8_t)(1 + ((own >> 5) & 3)) : 0;
				t.rgb[0] = t.rgb[1] = t.rgb[2] = word & 0xFFFFFF;
				static constexpr int Order[6] = { 0, 1, 2, 1, 3, 2 };
				for (int k = 0; k < 2; k++)
				{
					for (int j = 0; j < 3; j++)
						t.x[j] = xs[Order[k * 3 + j]], t.y[j] = ys[Order[k * 3 + j]], t.uv[j] = uvs[Order[k * 3 + j]];
					hudPolygons.push_back(t);
				}
				w += 9;
			}
			else if (code >= 0x40 && code < 0x44)
			{
				// a line: the split screen's, level, a pixel thick
				float x0, y0, x1, y1;
				corner(ramWord(base + 4), x0, y0), corner(ramWord(base + 8), x1, y1);
				const uint32_t rgb = word & 0xFFFFFF;
				hudLines.push_back({ { x0, x1 + 1, x0 }, { y0, y1, y0 + 1 }, { rgb, rgb, rgb }, { 0, 0, 0 } });
				hudLines.push_back({ { x1 + 1, x1 + 1, x0 }, { y1, y1 + 1, y0 + 1 }, { rgb, rgb, rgb }, { 0, 0, 0 } });
				w += 3;
			}
			else
			{
				static bool told = false;
				if (!told)
					std::fprintf(stderr, "deadstation rallycross: the HUD has a command %02x: by the list\n", code), told = true;
				return false;
			}
		}
		at = tag & 0x1FFFFF;
	}
	deadstation_replay_object(0);
	setTint(false);
	if (!hudLines.empty())
		deadstation_flat(hudLines.data(), hudLines.size());
	deadstation_triangles(hudPolygons.data(), hudPolygons.size());
	return true;
}

// Split screens natively. The two-player race's loop (0x80021xxx) walks the track twice (0x80014534, a sibling of 0x800117E4 with the same arguments), for a view each: 0x800217D8 the top, into the
// ordering table at +4208 of the frame's buffers, 0x8002198C the bottom, into +112. Between the two tables' lists: a fill for each view's background, and (+0x1DF50) what sets the bottom's area and offset;
// at the end of them (+0x1DFEC) what sets the whole screen's, then the line between the views and the HUD. So each view's walk here, its frame drawn with its own area and offset, and the list from there.
struct ViewFrame
{
	DeadstationWorldCamera camera {};
	std::vector<uint16_t> visible;
	std::vector<DeadstationWorldQuad> quads, quadsBlended, marks;
	std::vector<float> quadsBlendedZ;
	std::vector<DeadstationTriangle> solid[8], blended[8];
	std::vector<DeadstationGlow> depths[8], darkens[8];
	float viewZ[8] = {};
};
ViewFrame splitViews[2];
int splitWalked = 0;
uint32_t splitBuffers = 0;
void swapView(ViewFrame& v)
{
	std::swap(trackCamera, v.camera);
	trackVisible.swap(v.visible), trackQuads.swap(v.quads), trackQuadsBlended.swap(v.quadsBlended), trackMarks.swap(v.marks), trackQuadsBlendedZ.swap(v.quadsBlendedZ);
	for (int i = 0; i < 8; i++)
	{
		carSolid[i].swap(v.solid[i]), carBlended[i].swap(v.blended[i]), carDepths[i].swap(v.depths[i]), carDarkens[i].swap(v.darkens[i]);
		std::swap(carViewZ[i], v.viewZ[i]);
	}
}
int onSplitWalk(CPUState *cpu, uint32_t)
{
	const uint32_t ra = cpu->gpr[31];
	if (inTrack || !fullDetail || (ra != 0x800217D8 && ra != 0x8002198C))
		return 0;
	if (ra == 0x800217D8)
		splitBuffers = cpu->gpr[4] - 4208, splitWalked = 0;
	splitNative = true;
	const int took = onTrackDraw(cpu, 0);
	splitNative = false;
	if (took != 0 && splitWalked < 2)
		swapView(splitViews[splitWalked++]);
	return took;
}
// both views' frames; the list from what's left
bool emitSplit()
{
	if (splitWalked != 2)
		return false;
	for (int view = 0; view < 2; view++)
	{
		if (view == 1)
		{
			// (the game's own packet for the bottom's area and offset, on its own)
			const uint32_t at = splitBuffers + 0x1DF50, tag = guestRead(at);
			guestWord(at, (tag & 0xFF000000u) | 0xFFFFFFu);
			psx_native_gpu_list(at);
			guestWord(at, tag);
		}
		swapView(splitViews[view]);
		trackActive = true;
		emitFrame();
		swapView(splitViews[view]);
	}
	splitWalked = 0;
	trackOt = splitBuffers + 0x1DFEC;
	return true;
}
uint32_t drawListFrom(uint32_t ot)
{
	hudListRestore();
	if (!frameDrawn)
		return ot;
	frameDrawn = false;
	if (emitHud(trackOt))
		return 0;
	hudListHide(trackOt);
	return trackOt;
}

// the game draws the frame it built
void onBeforeDrawOTag()
{
	// (a race's frame: the track was walked; no car need be in view, from inside one looking away)
	const bool race = building.objects[0].present || (fullDetail && trackActive);
	deadstation_game_frames++;
	if (leavingRace && !race)
		leavingRace = false;
	if (!pages.built)
		buildPages();
	if (!race)
		routeMenus();
	drawingWorld.swap(buildingWorld);
	buildingWorld.clear();
	setTint(false);
	// the world's light this frame: the next strike of lightning, if it's that kind of night
	if (race)
	{
		weatherClock = blanksSeen;
		lightningNow = 0;
		if (look.lightning)
		{
			const uint32_t round = weatherClock / 400u, offset = mixBits(round * 7919u) % 300u, into = weatherClock % 400u;
			if (into >= offset && into < offset + 14u)
				lightningNow = (into - offset) % 6u < 2u ? 1.f : 0.45f;
		}
		tintColour[0] = tintBase[0], tintColour[1] = tintBase[1], tintColour[2] = tintBase[2];
		if (lightningNow > 0)
			for (int i = 0; i < 3; i++)
				tintColour[i] = std::min((int)((float)tintBase[i] * (1.f + 0.9f * lightningNow)), 511);
	}
	int32_t ox = 0, oy = 0;
	psx_gpu_draw_offset(&ox, &oy);
	building.drawX = (float)ox, building.drawY = (float)oy;
	sixtyRunning = sixty && race && guestByte(Players) <= 1;
	// (a frame of the game took most of the PlayStation's CPU for its two blanks: in one, and with the physics again, as a console overclocked)
	psx_cycle_rate = sixtyRunning || guestByte(Players) > 1 ? 96u : 256u;	// (split screens: two whole views didn't fit a PlayStation's CPU in its 30 frames a second, and at a fixed step that's slow motion)
	building.valid = race;
	deadstation_scene_perspective(race);
	deadstation_replay_frame(race && !sixtyRunning && !fullDetail ? &building : nullptr);
	drawingRanges.swap(buildingRanges);
	buildingRanges.clear();
	building = deadstation::replay::Frame {};
	currentObject = -1;
	if (fullDetail)
	{
		if (!emitSplit() && race && trackActive)
			emitFrame();
		return;
	}
	if (trackActive && trackSorted)
		trackActive = false;	// its list was walked: not again unless the game builds another
}
void onTriangleSource(uint32_t addr)
{
	const uint32_t a = addr & 0x1FFFFF;
	if (tintWanted || effectsDue)
	{
		bool world = false;
		for (const PacketSpan& span : drawingWorld)
			world = world || (a >= span.from && a < span.to);
		// past the world: what the weather and the night add, before what's drawn over it all
		if (!world && effectsDue && trackOpaqueDrawn)
		{
			setTint(false);
			emitEffects();
			currentObject = 0;
		}
		setTint(world);
	}
	int object = 0;
	for (const PacketRange& r : drawingRanges)
		if (a >= r.from && a < r.to)
		{
			object = r.object;
			break;
		}
	if (object != currentObject)
	{
		currentObject = object;
		deadstation_replay_object(object);
	}
}

// DEADSTATION_RALLYCROSS_FRAMES (tests): each frame the game hands the GPU, who from, and what's in it
uint32_t framesLogged = 0;
void onFrameLog(CPUState *cpu, uint32_t)
{
	uint32_t at = cpu->gpr[4] & 0x1FFFFC, tiles = 0, blackTiles = 0, fills = 0, sprites = 0, polys = 0, others = 0, nodes = 0;
	uint32_t firstBlack[4] = {};
	while (nodes < 40000)
	{
		const uint32_t word = guestRead(0x80000000u | at), length = word >> 24;
		nodes++;
		if (length != 0)
		{
			const uint32_t command = guestRead(0x80000000u | (at + 4));
			const uint32_t op = command >> 24;
			if (op == 0x02)
				fills++;
			else if ((op & 0xE0) == 0x60)
			{
				const bool textured = (op & 4) != 0;
				if (textured)
					sprites++;
				else
				{
					tiles++;
					if ((command & 0xFFFFFF) == 0)
					{
						if (blackTiles == 0)
						{
							firstBlack[0] = guestRead(0x80000000u | (at + 8));
							firstBlack[1] = op == 0x60 ? guestRead(0x80000000u | (at + 12)) : 0;
						}
						blackTiles++;
					}
				}
			}
			else if ((op & 0xE0) == 0x20 || (op & 0xE0) == 0x30)
				polys++;
			else
				others++;
		}
		const uint32_t next = word & 0xFFFFFF;
		if (next == 0xFFFFFF)
			break;
		at = next & 0x1FFFFC;
	}
	std::fprintf(stderr, "deadstation frame %u: from %08X sp %08X blanks %u: fills %u tiles %u black %u (%08X %08X) sprites %u polys %u others %u nodes %u\n", framesLogged++, cpu->gpr[31], cpu->gpr[29], guestRead(BlankCount), fills, tiles, blackTiles, firstBlack[0], firstBlack[1], sprites, polys, others, nodes);
}

struct Registration
{
	Registration()
	{
		if (std::getenv("DEADSTATION_RALLYCROSS_FRAMES") != nullptr)
			psx_mod_register_function_entry_plugin(Id, 0x800489F8, onFrameLog);
		psx_mod_register_function_filter_plugin(Id, Height, onHeightNative);
		psx_mod_register_function_filter_plugin(Id, GridSample, onGridSample);
		psx_mod_register_function_entry_plugin(Id, RaceLoad, onRaceLoad);
		psx_mod_register_function_entry_plugin(Id, ImageUpload, onImageUpload);
		stats = std::getenv("DEADSTATION_RALLYCROSS_STATS") != nullptr;
		autopilot = std::getenv("DEADSTATION_RALLYCROSS_AUTOPILOT") != nullptr;
		hudLog = std::getenv("DEADSTATION_RALLYCROSS_HUDLOG") != nullptr;
		menusAtSixty = std::getenv("DEADSTATION_RALLYCROSS_MENU30") == nullptr;
		if (const char *kick = std::getenv("DEADSTATION_RALLYCROSS_KICK"))
			kickPercent = std::atoi(kick);
		secondPadForTests = std::getenv("DEADSTATION_RALLYCROSS_SECOND_PAD") != nullptr;
		// (=N: from that step on, for a whole race: what only exists moving doesn't show on a held view)
		if (const char *cost = std::getenv("DEADSTATION_RALLYCROSS_COST"))
		{
			costing = true;
			costStep = std::atoi(cost);
		}
		if (autopilot)
			psx_mod_register_function_filter_plugin(Id, 0x8003C4A0, onLastLapHandover);
		if (const char *freeze = std::getenv("DEADSTATION_RALLYCROSS_FREEZE"))
			freezeAfter = (uint32_t)std::strtoul(freeze, nullptr, 10);
		psx_mod_register_vblank_plugin(Id, onVBlank);
		// the PlayStation SDK as host code (sdk/brick/psx_sdk.h): Rally Cross's copies (PsyQ 3.6.1, signature-matched)
		PsxSdkAddresses sdk;
		sdk.drawOTag = 0x800489F8;		// LIBGPU SYS.OBJ
		sdk.loadImage = 0x800486BC;
		sdk.moveImage = 0x80048784;
		const char *sdkOff = std::getenv("DEADSTATION_SDK_OFF");
		sixty = sdkOff == nullptr || std::strstr(sdkOff, "sixty") == nullptr;
		if (sixty)
		{
			psx_mod_register_function_filter_plugin(Id, Multiply, onMultiply);
			helpersNative = !(sdkOff != nullptr && std::strstr(sdkOff, "helpers") != nullptr);
			if (helpersNative)
			{
				psx_mod_register_function_filter_plugin(Id, 0x8003F4DC, onDot);
				psx_mod_register_function_filter_plugin(Id, 0x8003F564, onRoot);
				psx_mod_register_function_filter_plugin(Id, 0x8003CA54, onRoughLength);
				psx_mod_register_function_filter_plugin(Id, 0x8004B0BC, onVectorNormal);
				psx_mod_register_function_filter_plugin(Id, 0x8004B0E8, onVectorNormalS);
			}
			psx_mod_register_function_filter_plugin(Id, PhysicsStep, onPhysicsStep);
			psx_mod_register_function_filter_plugin(Id, CarsSetUp, onCarsSetUp);
			psx_mod_register_function_filter_plugin(Id, PadsRead, onPadsRead);
			psx_mod_register_function_filter_plugin(Id, RaceBookkeeping, onBookkeeping);
			psx_mod_register_function_filter_plugin(Id, 0x8003F7AC, everySecondPass<0x80020440>);
			psx_mod_register_function_filter_plugin(Id, 0x80040184, everySecondPass<0x80020448>);
			psx_mod_register_function_filter_plugin(Id, Brains, onBrains);
			psx_mod_register_function_filter_plugin(Id, CameraWall, onCameraWall);
			psx_mod_register_function_entry_plugin(Id, CameraSwing, onCameraSwing);
			psx_mod_register_function_filter_plugin(Id, WorldEffects, onWorldEffects);
			psx_mod_register_function_filter_plugin(Id, DrawDone, onDrawDone);
			psx_mod_register_function_filter_plugin(Id, VSync, onVSync);
			psx_mod_register_function_entry_plugin(Id, AfterTheWait, onAfterTheWait);
			psx_mod_register_function_filter_plugin(Id, 0x80023A68, everySecondPass<0x800204B4>);
			psx_mod_register_function_filter_plugin(Id, 0x80038D70, everySecondPass<0x800206E4>);
		}
		else if (autopilot)
			psx_mod_register_function_filter_plugin(Id, CarsSetUp, onCarsSetUp);
		if (sdkOff == nullptr || std::strstr(sdkOff, "replay") == nullptr)
		{
			sdk.beforeDrawOTag = onBeforeDrawOTag;
			sdk.drawOTagFrom = drawListFrom;
			psx_mod_register_function_filter_plugin(Id, 0x80014534, onSplitWalk);
			psx_mod_register_function_filter_plugin(Id, CarRenderer, onCarRenderer);
			if (sdkOff != nullptr && std::strstr(sdkOff, "drawin") != nullptr)
				psx_gpu_triangle_source_hook = onTriangleSource;
		}
		if (sdkOff == nullptr || std::strstr(sdkOff, "track") == nullptr)
		{
			psx_mod_register_function_filter_plugin(Id, TrackDraw, onTrackDraw);
			fullDetail = sdkOff == nullptr || std::strstr(sdkOff, "drawin") == nullptr;
			if (fullDetail)
			{
				psx_mod_register_function_filter_plugin(Id, CarDepthCall, onCarDepth);
				psx_mod_register_function_filter_plugin(Id, CarPolygonsDraw, onCarPolygons);
			}
			if (!fullDetail)
				psx_gpu_list_node_hook = onListNode;
			else
				psx_mod_register_function_filter_plugin(Id, 0x8002C378, onGameSky);
		}
		// VSync's wait not native (v_wait 0x80046784 on the counter 0x8005A2C4): Rally Cross waits two blanks a frame, and
		// after the native jump to the first the rest of its poll ran out of polls (32768, a count) before the second
		// under the lean timing: LIBETC's timeout path, ~1 fps. (PSXRecomp's idle skip on that loop gained nothing: the
		// pads' and the CD's events, ~60K a second, keep each skip short; native pads and CD first.)
		if (std::getenv("DEADSTATION_SDK") == nullptr || std::strcmp(std::getenv("DEADSTATION_SDK"), "emulated") != 0)
			psx_sdk_install(Id, sdk);
		const char *off = std::getenv("DEADSTATION_SDK_OFF");
		if ((std::getenv("DEADSTATION_SDK") == nullptr || std::strcmp(std::getenv("DEADSTATION_SDK"), "emulated") != 0) &&
				(off == nullptr || std::strstr(off, "pad") == nullptr))
			psx_mod_register_function_filter_plugin(Id, PadRead, onPadRead);
		psx_mod_register_function_filter_plugin(Id, AiHands, onAiHands);
		psx_mod_register_function_filter_plugin(Id, CameraMove, onCameraMove);
		psx_mod_register_function_filter_plugin(Id, MenuText, onMenuText);
		psx_mod_register_function_entry_plugin(Id, MenuPanel, onMenuPanel);
		psx_mod_register_function_entry_plugin(Id, CdStop, onCdStop);
		if (const char *drive = std::getenv("DEADSTATION_RALLYCROSS_DRIVE"))
		{
			for (const char *at = drive; *at != 0;)
			{
				float from = 0, to = 0;
				unsigned mask = 0;
				int used = 0;
				if (std::sscanf(at, "%f-%f:%x%n", &from, &to, &mask, &used) != 3)
					break;
				handsDrive.push_back({ from, to, (uint16_t)mask });
				at += used;
				if (*at == ',')
					at++;
			}
			handsTest = true;
		}
		if (const char *trials = std::getenv("DEADSTATION_RALLYCROSS_ROLL"))
		{
			rollTrials = std::atoi(trials);
			handsTest = rollTrials > 0 || handsTest;
		}
		if (handsTest && !sixty)
		{
			psx_mod_register_function_filter_plugin(Id, PhysicsStep, onPhysicsStep);
			psx_mod_register_function_filter_plugin(Id, Brains, onBrains);
		}
	}
} registration;
}

// Resolution: the game's frames drawn by the presenter's Vulkan at this size, the display scaling it to the panel; VRAM at twice
// the PlayStation's but at 320 x 240. (A test's DEADSTATION_RESOLUTION comes first.)
// The headlights in the world's shaders: this game's own, in place of the SDK's plain ones
extern "C" void deadstation_module_start(void)
{
	const DeadstationWorldShaders shaders = { world_vert, sizeof(world_vert), world_lit_vert, sizeof(world_lit_vert), world_lit_frag, sizeof(world_lit_frag), world_edge_lit_frag, sizeof(world_edge_lit_frag) };
	deadstation_world_shaders(&shaders);
}

extern "C" void deadstation_module_setting(const char *key, const char *value)
{
	if (std::strcmp(key, "scale") != 0)
		return;
	setenv("DEADSTATION_RESOLUTION", value, 0);
	if (std::strcmp(getenv("DEADSTATION_RESOLUTION"), "320") != 0)
		setenv("PSX_INTERNAL_RESOLUTION", "480", 0);
}
