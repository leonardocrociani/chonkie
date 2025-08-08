import React, { useEffect, useState } from "react";
import { Update } from "@mintlify/components";

export const GithubReleases = () => {
  const [releases, setReleases] = useState([]);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState(null);

  useEffect(() => {
    fetch("https://api.github.com/repos/chonkie-inc/chonkie/releases")
      .then((res) => {
        if (!res.ok) throw new Error("Failed to fetch releases");
        return res.json();
      })
      .then((data) => {
        setReleases(data);
        setLoading(false);
      })
      .catch((err) => {
        setError(err.message);
        setLoading(false);
      });
  }, []);

  return (
    <div className="p-4 border dark:border-zinc-950/80 rounded-xl not-prose">
      <h2 className="font-bold mb-2">GitHub Releases</h2>
      {loading && <div>Loading...</div>}
      {error && <div className="text-red-500">Error: {error}</div>}
      {!loading && !error && (
        <div>
          {releases.map((release) => (
            <Update
              key={release.id}
              label={new Date(release.published_at).toLocaleString("default", { month: "long", year: "numeric" })}
              description={release.name}
            >
              {release.body}
            </Update>
          ))}
        </div>
      )}
    </div>
  );
};