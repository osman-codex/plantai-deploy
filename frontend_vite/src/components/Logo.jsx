/**
 * PlantGuard mark: a leaf whose veins form a neural network.
 * Pure inline SVG — scales crisply, no asset files.
 */
export default function Logo({ size = 34, id = "pg" }) {
  const g = `pg-leaf-${id}`;
  return (
    <svg
      width={size}
      height={size}
      viewBox="0 0 48 48"
      fill="none"
      aria-hidden="true"
      style={{ display: "block" }}
    >
      <defs>
        <linearGradient id={g} x1="8" y1="4" x2="40" y2="44" gradientUnits="userSpaceOnUse">
          <stop offset="0" stopColor="#8CD64F" />
          <stop offset="0.55" stopColor="#4A9A2A" />
          <stop offset="1" stopColor="#1E3D0F" />
        </linearGradient>
      </defs>

      {/* leaf body */}
      <path
        d="M41 7C22.5 7.5 9.5 15.5 9.5 30.5c0 5.2 2 8.6 3.6 10.1C22 40.5 41 34 41 7Z"
        fill={`url(#${g})`}
      />
      {/* leaf outline highlight */}
      <path
        d="M41 7C22.5 7.5 9.5 15.5 9.5 30.5c0 5.2 2 8.6 3.6 10.1C22 40.5 41 34 41 7Z"
        stroke="#EAF6E0"
        strokeOpacity="0.35"
        strokeWidth="1"
      />

      {/* midrib (stem → tip) */}
      <path
        d="M12.5 40.5C17 30 26 19 38.5 9.5"
        stroke="#EAF6E0"
        strokeWidth="2.1"
        strokeLinecap="round"
      />

      {/* neural connections (veins) */}
      <path
        d="M23.5 23.5 15 31.5M23.5 23.5l9-6.5M23.5 23.5l7.5 7M31 9.8l1.5 7.2M19 36.8l4.5-6.3"
        stroke="#DFF3CE"
        strokeWidth="1.3"
        strokeLinecap="round"
        opacity="0.85"
      />

      {/* neural nodes */}
      <circle cx="23.5" cy="23.5" r="2.7" fill="#F2FAE8" />
      <circle cx="23.5" cy="23.5" r="1.1" fill="#2D5016" />
      <circle cx="32.5" cy="17" r="2" fill="#F2FAE8" />
      <circle cx="15" cy="31.5" r="2" fill="#F2FAE8" />
      <circle cx="31" cy="30.5" r="1.6" fill="#CFE9B6" />
      <circle cx="31" cy="9.8" r="1.5" fill="#F2FAE8" />
      <circle cx="19" cy="36.8" r="1.5" fill="#CFE9B6" />
    </svg>
  );
}
