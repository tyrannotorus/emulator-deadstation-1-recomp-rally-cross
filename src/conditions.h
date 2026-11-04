// Rally Cross's race conditions: what a season, a time of day and the weather make of a track. Tables and sums only, nothing of the
// game's (the plugin applies them; a host tool can preview them), so the art can be changed here alone.
//
// Season: a track's ground and scenery are 8-bit textures through one 256-colour palette, loaded as the last row of its first texture
// file, so a season is that one row recoloured as it loads. Each colour is taken for foliage, earth, rock, snow or other by its hue and
// saturation; other (signs, spectators' clothes) is left alone.
//
// Original / Day / Clear is the game exactly as it shipped.
#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>

namespace conditions
{
enum Track : uint8_t { Oasis, Islands, LostMines, Alpine, Gardens, Stadium, TrackCount };
enum Season : uint8_t { Original, Spring, Summer, Autumn, Winter, SeasonCount };
enum Time : uint8_t { Day, Dawn, Dusk, Night, TimeCount };
enum Weather : uint8_t { Clear, Rain, Storm, Fog, Snow, WeatherCount };
enum Falling : uint8_t { Nothing, Drizzle, Downpour, Flakes, Sand };

struct Rgb
{
	float r, g, b;
};
inline Rgb operator*(const Rgb& a, const Rgb& b) { return { a.r * b.r, a.g * b.g, a.b * b.b }; }
inline Rgb mix(const Rgb& a, const Rgb& b, float k) { return { a.r + (b.r - a.r) * k, a.g + (b.g - a.g) * k, a.b + (b.b - a.b) * k }; }

// what becomes of one kind of colour: its hue turned towards hue by turn (0 none, 1 all the way), its saturation and value scaled
struct Shift
{
	float hue, turn, sat, val;
};
constexpr Shift Keep = { 0, 0, 1, 1 };
struct SeasonLook
{
	Shift foliage, earth, rock, snow;
	float whiten;			// snow lying on foliage, earth and rock: 0 none, 1 all white
	float snowAbove;		// colours of snow at or above this value stay snow whatever its shift
	Rgb light;				// the season's light, over the time of day's
};
// by track, then Spring, Summer, Autumn, Winter (Original touches nothing)
constexpr SeasonLook Seasons[TrackCount][4] = {
	// Oasis: sand and palms; never much rain. A green flush, baking heat, amber haze, frost at dawn.
	{ { { 95, 0.7f, 1.2f, 1.08f }, { 62, 0.2f, 0.85f, 1.02f }, Keep, Keep, 0, 2, { 1, 1, 0.97f } },
	  { { 70, 0.3f, 0.85f, 1.0f }, { 38, 0.2f, 0.9f, 1.1f }, { 40, 0.2f, 1, 1.05f }, Keep, 0, 2, { 1.06f, 1.02f, 0.94f } },
	  { { 38, 0.8f, 1.1f, 0.95f }, { 22, 0.45f, 1.2f, 0.92f }, Keep, Keep, 0, 2, { 1.06f, 0.94f, 0.82f } },
	  { { 110, 0.3f, 0.5f, 0.85f }, { 40, 0, 0.45f, 1.05f }, { 220, 0.5f, 1, 1 }, Keep, 0.38f, 2, { 0.92f, 0.96f, 1.08f } } },
	// Islands: jungle over dark earth. Lush, deep, dry gold, a cool wet season (no snow).
	{ { { 112, 0.5f, 1.3f, 1.2f }, { 28, 0.1f, 1.05f, 1.05f }, Keep, Keep, 0, 2, { 1, 1.02f, 0.98f } },
	  { { 90, 0.3f, 1.15f, 1.05f }, { 30, 0.1f, 1.0f, 1.1f }, Keep, Keep, 0, 2, { 1.05f, 1.02f, 0.95f } },
	  { { 48, 0.75f, 1.15f, 1.05f }, { 28, 0.3f, 1.1f, 1.0f }, Keep, Keep, 0, 2, { 1.05f, 0.97f, 0.85f } },
	  { { 140, 0.35f, 0.65f, 0.78f }, { 25, 0, 0.7f, 0.72f }, { 210, 0.3f, 1, 0.85f }, Keep, 0, 2, { 0.86f, 0.94f, 1.02f } } },
	// Lost Mines: timber, rock and moss. Fresh moss, dust, rust and gold, snow on the ridges.
	{ { { 105, 0.6f, 1.3f, 1.18f }, { 30, 0.1f, 0.95f, 0.95f }, Keep, Keep, 0, 2, { 0.98f, 1, 1 } },
	  { { 75, 0.4f, 0.95f, 1.05f }, { 34, 0.2f, 1.0f, 1.1f }, { 36, 0.15f, 1, 1.05f }, Keep, 0, 2, { 1.06f, 1.0f, 0.92f } },
	  { { 26, 0.9f, 1.4f, 1.05f }, { 24, 0.3f, 1.2f, 0.95f }, { 30, 0.15f, 1, 1 }, Keep, 0, 2, { 1.05f, 0.93f, 0.8f } },
	  { { 120, 0.5f, 0.35f, 0.7f }, { 30, 0, 0.45f, 1.0f }, { 215, 0.4f, 1, 1 }, Keep, 0.8f, 2, { 0.9f, 0.96f, 1.1f } } },
	// Alpine: all snow as it shipped. Melting, meadows, golden grass, and Winter a colder blue of what it was. The brightest keeps its snow.
	{ { { 100, 0.6f, 1.3f, 1.1f }, { 30, 0, 1, 0.95f }, Keep, { 98, 1, 0.55f, 0.6f }, 0, 0.86f, { 1, 1, 0.98f } },
	  { { 95, 0.5f, 1.3f, 1.1f }, { 32, 0, 1, 1 }, Keep, { 92, 1, 0.68f, 0.62f }, 0, 2, { 1.05f, 1.02f, 0.95f } },
	  { { 40, 0.6f, 1.2f, 1.0f }, { 28, 0.1f, 1.1f, 0.95f }, Keep, { 42, 1, 0.65f, 0.72f }, 0, 0.96f, { 1.05f, 0.95f, 0.84f } },
	  { Keep, Keep, Keep, { 225, 0.4f, 1.2f, 0.98f }, 0, 2, { 0.92f, 0.96f, 1.1f } } },
	// Gardens: brick, sandstone and clipped hedges. Blossom green, full leaf, red and orange hedges, snow on them all.
	{ { { 108, 0.6f, 1.35f, 1.25f }, Keep, Keep, Keep, 0, 2, { 1, 1, 0.99f } },
	  { { 90, 0.3f, 1.2f, 1.1f }, { 34, 0, 0.95f, 1.08f }, Keep, Keep, 0, 2, { 1.05f, 1.02f, 0.95f } },
	  { { 20, 0.95f, 1.7f, 1.25f }, { 26, 0.1f, 1.05f, 0.97f }, Keep, Keep, 0, 2, { 1.05f, 0.95f, 0.84f } },
	  { { 130, 0.4f, 0.4f, 0.85f }, { 30, 0, 0.75f, 1.02f }, { 215, 0.4f, 1, 1.05f }, Keep, 0.72f, 2, { 0.9f, 0.95f, 1.08f } } },
	// Stadium: under a roof. The season doesn't reach it.
	{ { Keep, Keep, Keep, Keep, 0, 2, { 1, 1, 1 } }, { Keep, Keep, Keep, Keep, 0, 2, { 1, 1, 1 } },
	  { Keep, Keep, Keep, Keep, 0, 2, { 1, 1, 1 } }, { Keep, Keep, Keep, Keep, 0, 2, { 1, 1, 1 } } },
};

inline void toHsv(float r, float g, float b, float& h, float& s, float& v)
{
	const float hi = std::max({ r, g, b }), lo = std::min({ r, g, b }), d = hi - lo;
	v = hi;
	s = hi > 0 ? d / hi : 0;
	h = 0;
	if (d > 0)
	{
		h = hi == r ? (g - b) / d : hi == g ? 2 + (b - r) / d : 4 + (r - g) / d;
		h *= 60;
		if (h < 0)
			h += 360;
	}
}
inline void fromHsv(float h, float s, float v, float& r, float& g, float& b)
{
	const float c = v * s, x = c * (1 - std::fabs(std::fmod(h / 60, 2.f) - 1)), m = v - c;
	const int i = (int)(h / 60) % 6;
	r = (i == 0 || i == 5 ? c : i == 1 || i == 4 ? x : 0) + m;
	g = (i == 1 || i == 2 ? c : i == 0 || i == 3 ? x : 0) + m;
	b = (i == 3 || i == 4 ? c : i == 2 || i == 5 ? x : 0) + m;
}

// one palette colour (the PlayStation's 15 bits and its STP bit). All zero is transparent, and stays so.
inline uint16_t seasonColour(uint16_t c, const SeasonLook& look)
{
	if (c == 0)
		return c;
	float r = (float)(c & 31) / 31, g = (float)((c >> 5) & 31) / 31, b = (float)((c >> 10) & 31) / 31, h, s, v;
	toHsv(r, g, b, h, s, v);
	const Shift *shift = nullptr;
	bool ground = true;
	if (h >= 185 && h <= 270 && s < 0.45f && v > 0.42f)
	{
		// (Alpine's snow, a pale blue; on the others a pale blue is rare and takes Keep)
		shift = v >= look.snowAbove ? &Keep : &look.snow;
		ground = false;
	}
	else if (s < 0.14f)
		shift = &look.rock;
	else if (h >= 58 && h <= 175)
		shift = &look.foliage;
	else if (h >= 12 && h < 58 && s < 0.82f)
		shift = &look.earth;
	else if (look.whiten > 0 && v < 0.42f)
		shift = &Keep;		// (the darks of everything else: snow lies on them too, or they speckle it)
	else
		return c;
	float d = shift->hue - h;
	if (d > 180)
		d -= 360;
	else if (d < -180)
		d += 360;
	h += d * shift->turn;
	if (h < 0)
		h += 360;
	else if (h >= 360)
		h -= 360;
	s = std::min(s * shift->sat, 1.f), v = std::min(v * shift->val, 1.f);
	fromHsv(h, s, v, r, g, b);
	if (ground && look.whiten > 0)
	{
		const float snow = std::min(0.76f + 0.34f * v, 1.f);
		r += (0.88f * snow - r) * look.whiten, g += (0.92f * snow - g) * look.whiten, b += (1.0f * snow - b) * look.whiten;
	}
	const auto bits = [](float x) { return (uint16_t)std::clamp((int)std::lround(x * 31), 0, 31); };
	uint16_t out = (uint16_t)(bits(r) | bits(g) << 5 | bits(b) << 10 | (c & 0x8000));
	return out == 0 ? (uint16_t)0x0421 : out;	// (never made transparent)
}

// What a race looks like, all told
struct Look
{
	Rgb world;				// over the track, its scenery and the cars
	bool skyChanged;
	Rgb zenith, horizon;	// the sky's gradient
	Rgb panorama;			// over the mountains and clouds round the horizon (1: as they are)
	bool hidePanorama;		// lost in the weather
	double sun[3];			// the way the sun (or the moon) shines (the game's y points down)
	int shadowGrey;			// a shadow's colour (blended half way to it): 0 under the sun; the soft one under an overcast sky lighter
	bool shadows;
	bool headlights;
	Rgb fog;
	float fogNear, fogFar;	// fogFar 0: none
	int fogVeils;			// how many between them
	Falling falling;
	bool lightning;
	bool stars;
	float beam;				// how strong headlights are on the world
	float haze;				// how much of their light hangs in the air
};

inline Look lookFor(Track track, Season season, Time time, Weather weather)
{
	Look l {};
	l.world = { 1, 1, 1 };
	l.panorama = { 1, 1, 1 };
	l.sun[0] = 0.85, l.sun[1] = 1.0, l.sun[2] = 0.5;
	l.shadowGrey = 0;
	l.shadows = true;
	if (season != Original)
		l.world = Seasons[track][season - 1].light;
	switch (time)
	{
	case Day:
		break;
	case Dawn:
		l.world = l.world * Rgb { 1.0f, 0.82f, 0.76f };
		l.skyChanged = true;
		l.zenith = { 0.26f, 0.34f, 0.58f }, l.horizon = { 0.98f, 0.70f, 0.56f };
		l.panorama = { 0.92f, 0.72f, 0.66f };
		l.sun[0] = -1.7, l.sun[2] = 0.45;
		break;
	case Dusk:
		l.world = l.world * Rgb { 1.05f, 0.68f, 0.5f };
		l.skyChanged = true;
		l.zenith = { 0.2f, 0.17f, 0.42f }, l.horizon = { 1.0f, 0.5f, 0.24f };
		l.panorama = { 0.82f, 0.5f, 0.42f };
		l.sun[0] = 1.8, l.sun[2] = 0.7;
		l.headlights = true;
		break;
	case Night:
		l.world = l.world * Rgb { 0.2f, 0.25f, 0.4f };
		l.skyChanged = true;
		l.zenith = { 0.012f, 0.018f, 0.055f }, l.horizon = { 0.05f, 0.07f, 0.15f };
		l.panorama = { 0.12f, 0.15f, 0.26f };
		l.shadows = false;
		l.headlights = true;
		l.stars = weather == Clear;
		break;
	default:
		break;
	}
	const auto overcast = [&](const Rgb& sky, float amount, const Rgb& light) {
		const Rgb z = l.skyChanged ? l.zenith : sky, h = l.skyChanged ? l.horizon : sky;
		l.skyChanged = true;
		l.zenith = mix(z, sky * Rgb { 0.82f, 0.82f, 0.82f }, amount), l.horizon = mix(h, sky, amount);
		l.panorama = mix(l.panorama, sky, amount * 0.85f);
		l.world = l.world * light;
		// (no sun to cast it, but the sky's light doesn't reach under a car: right under it, and softer)
		l.sun[0] = 0.14, l.sun[1] = 1.0, l.sun[2] = 0.1;
		l.shadowGrey = 64;
	};
	const float dim = time == Night ? 0.14f : time == Dawn || time == Dusk ? 0.55f : 1.f;
	switch (weather)
	{
	case Clear:
		break;
	case Rain:
		overcast(Rgb { 0.47f, 0.5f, 0.56f } * Rgb { dim, dim, dim }, 0.9f, { 0.72f, 0.76f, 0.84f });
		l.fog = Rgb { 0.42f, 0.45f, 0.5f } * Rgb { dim, dim, dim }, l.fogNear = 1800, l.fogFar = 11000, l.fogVeils = 4;
		l.falling = Drizzle;
		break;
	case Storm:
		overcast(track == Oasis ? Rgb { 0.62f, 0.44f, 0.26f } * Rgb { dim, dim, dim } : Rgb { 0.19f, 0.21f, 0.25f } * Rgb { dim, dim, dim }, 0.95f,
				track == Oasis ? Rgb { 0.82f, 0.68f, 0.5f } : Rgb { 0.5f, 0.55f, 0.66f });
		l.fog = track == Oasis ? Rgb { 0.6f, 0.43f, 0.26f } * Rgb { dim, dim, dim } : Rgb { 0.2f, 0.22f, 0.27f } * Rgb { dim, dim, dim };
		l.fogNear = 1000, l.fogFar = track == Oasis ? 4500 : 8000, l.fogVeils = track == Oasis ? 9 : 4;
		l.falling = track == Oasis ? Sand : Downpour;
		l.hidePanorama = track == Oasis;
		l.lightning = track != Oasis;
		l.headlights = true;
		break;
	case Fog:
		overcast(Rgb { 0.68f, 0.7f, 0.72f } * Rgb { dim, dim, dim }, 1.f, { 0.82f, 0.84f, 0.86f });
		l.hidePanorama = true;
		l.fog = Rgb { 0.68f, 0.7f, 0.72f } * Rgb { dim, dim, dim }, l.fogNear = 500, l.fogFar = 4200, l.fogVeils = 9;
		l.headlights = true;
		break;
	case Snow:
		overcast(Rgb { 0.76f, 0.79f, 0.84f } * Rgb { dim, dim, dim }, 0.9f, { 0.84f, 0.88f, 0.96f });
		l.fog = Rgb { 0.78f, 0.81f, 0.86f } * Rgb { dim, dim, dim }, l.fogNear = 1200, l.fogFar = 7000, l.fogVeils = 5;
		l.falling = Flakes;
		break;
	default:
		break;
	}
	// indoors: no sky to change, and the time of day is how far down the lights are
	if (track == Stadium)
	{
		l.skyChanged = false;
		l.hidePanorama = false;
		l.panorama = { 1, 1, 1 };
		l.fogFar = 0;
		l.falling = Nothing;
		l.lightning = l.stars = false;
		if (time == Night)
			l.world = { 0.3f, 0.32f, 0.42f };
		l.shadows = time != Night;
		l.sun[0] = 0.25, l.sun[2] = 0.2;
	}
	l.beam = time == Night ? 1.f : time == Dusk ? 0.5f : 0.65f;
	l.haze = l.fogFar > 0 ? (weather == Fog || weather == Storm ? 1.f : 0.6f) : 0.28f;
	return l;
}

// Season mode: the year goes by over the races and the going gets harder with it. It opens by day in the clear, has a night race among its last, and between them the light goes and the
// weather worsens, each race's its own and the same every time it's run. Each track keeps to what its climate allows: never snow on the desert or the islands, never at all on the others out of winter.
struct Conditions
{
	Season season;
	Time time;
	Weather weather;
};
inline Conditions calendar(Track track, int race, int races, int version, int backward, int style)
{
	const int last = std::max(races, 1) - 1;
	// (the first race is the year's first, the last its last, whatever their number)
	const int part = last > 0 ? std::clamp(race * 4 / last, 0, 3) : 0;
	// (race + 1: the first race on the first track would hash to nothing at all, and so the same every time)
	uint32_t h = (uint32_t)(race + 1) * 2654435761u + (uint32_t)track * 40503u + (uint32_t)version * 977u + (uint32_t)backward * 131u;
	h ^= h >> 15, h *= 0x2C1B3C6Du, h ^= h >> 13;
	const unsigned hour = h & 7, w = (h >> 8) & 15;
	const bool snowy = track == Alpine || track == LostMines || track == Gardens;
	// The season's night race, one of its last, winter's or the last two, by the season alone so it's the one race whichever is being run.
	int closing = 2;
	while (last > 0 && closing < last && std::clamp((last - closing) * 4 / last, 0, 3) == 3)
		closing++;
	closing = std::min(closing, std::max(last, 1));
	uint32_t whole = (uint32_t)races * 2654435761u + (uint32_t)style * 40503u;
	whole ^= whole >> 15, whole *= 0x2C1B3C6Du, whole ^= whole >> 13;
	const int nightRace = last - (int)(whole % (uint32_t)closing);
	Conditions c { (Season)(Spring + part), Day, Clear };
	switch (part)
	{
	case 0:
		c.weather = race > 0 && w < 3 ? Rain : Clear;
		break;
	case 1:
		c.time = hour < 6 ? Day : hour == 6 ? Dawn : Dusk;
		c.weather = w < 4 ? Rain : Clear;
		break;
	case 2:
		c.time = hour < 3 ? Day : hour < 5 ? Dawn : Dusk;
		c.weather = w < 5 ? Rain : w < 8 ? Fog : w < 10 ? Storm : Clear;
		break;
	default:
		c.time = Dusk;
		c.weather = snowy ? (w < 11 ? Snow : w < 13 ? Fog : Clear) : w < 6 ? Storm : w < 10 ? Rain : w < 13 ? Fog : Clear;
		break;
	}
	if (race == nightRace && last > 0)
		c.time = Night;
	if (track == Oasis && c.weather == Rain)
		c.weather = Clear;	// (it doesn't rain on the desert: its Storm is a sandstorm)
	if (track == Stadium)
		c.weather = Clear;
	return c;
}
}
