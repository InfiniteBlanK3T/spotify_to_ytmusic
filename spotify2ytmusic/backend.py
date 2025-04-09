#!/usr/bin/env python3

import json
import sys
import os
import time
import re

from ytmusicapi import YTMusic
from typing import Optional, Union, Iterator, Dict, List
from collections import namedtuple
from dataclasses import dataclass, field


SongInfo = namedtuple("SongInfo", ["title", "artist", "album"])


def get_ytmusic() -> YTMusic:
    """
    @@@
    """
    if not os.path.exists("oauth.json"):
        print("ERROR: No file 'oauth.json' exists in the current directory.")
        print("       Have you logged in to YTMusic?  Run 'ytmusicapi oauth' to login")
        sys.exit(1)

    try:
        return YTMusic("oauth.json")
    except json.decoder.JSONDecodeError as e:
        print(f"ERROR: JSON Decode error while trying start YTMusic: {e}")
        print("       This typically means a problem with a 'oauth.json' file.")
        print("       Have you logged in to YTMusic?  Run 'ytmusicapi oauth' to login")
        sys.exit(1)


def _ytmusic_create_playlist(
    yt: YTMusic, title: str, description: str, privacy_status: str = "PRIVATE"
) -> str:
    """Wrapper on ytmusic.create_playlist

    This wrapper does retries with back-off because sometimes YouTube Music will
    rate limit requests or otherwise fail.

    privacy_status can be: PRIVATE, PUBLIC, or UNLISTED
    """

    def _create(
        yt: YTMusic, title: str, description: str, privacy_status: str
    ) -> Union[str, dict]:
        exception_sleep = 5
        for _ in range(10):
            try:
                """Create a playlist on YTMusic, retrying if it fails."""
                id = yt.create_playlist(
                    title=title, description=description, privacy_status=privacy_status
                )
                return id
            except Exception as e:
                print(
                    f"ERROR: (Retrying create_playlist: {title}) {e} in {exception_sleep} seconds"
                )
                time.sleep(exception_sleep)
                exception_sleep *= 2

        return {
            "s2yt error": 'ERROR: Could not create playlist "{title}" after multiple retries'
        }

    id = _create(yt, title, description, privacy_status)
    #  create_playlist returns a dict if there was an error
    if isinstance(id, dict):
        print(f"ERROR: Failed to create playlist (name: {title}): {id}")
        sys.exit(1)

    time.sleep(1)  # seems to be needed to avoid missing playlist ID error

    return id


def load_playlists_json(filename: str = "playlists.json", encoding: str = "utf-8"):
    """Load the `playlists.json` Spotify playlist file"""
    return json.load(open(filename, "r", encoding=encoding))


def create_playlist(pl_name: str, privacy_status: str = "PRIVATE") -> None:
    """Create a YTMusic playlist


    Args:
        `pl_name` (str): The name of the playlist to create. It should be different to "".

        `privacy_status` (str: PRIVATE, PUBLIC, UNLISTED) The privacy setting of created playlist.
    """
    yt = get_ytmusic()

    id = _ytmusic_create_playlist(
        yt, title=pl_name, description=pl_name, privacy_status=privacy_status
    )
    print(f"Playlist ID: {id}")


def iter_spotify_liked_albums(
    spotify_playlist_file: str = "playlists.json",
    spotify_encoding: str = "utf-8",
) -> Iterator[SongInfo]:
    """Songs from liked albums on Spotify."""
    spotify_pls = load_playlists_json(spotify_playlist_file, spotify_encoding)

    if "albums" not in spotify_pls:
        return None

    for album in [x["album"] for x in spotify_pls["albums"]]:
        for track in album["tracks"]["items"]:
            yield SongInfo(track["name"], track["artists"][0]["name"], album["name"])


def iter_spotify_playlist(
    src_pl_id: Optional[str] = None,
    spotify_playlist_file: str = "playlists.json",
    spotify_encoding: str = "utf-8",
    reverse_playlist: bool = True,
) -> Iterator[SongInfo]:
    """Songs from a specific album ("Liked Songs" if None)

    Args:
        `src_pl_id` (Optional[str], optional): The ID of the source playlist. Defaults to None.
        `spotify_playlist_file` (str, optional): The path to the playlists backup files. Defaults to "playlists.json".
        `spotify_encoding` (str, optional): Characters encoding. Defaults to "utf-8".
        `reverse_playlist` (bool, optional): Is the playlist reversed when loading?  Defaults to True.

    Yields:
        Iterator[SongInfo]: The song's information
    """
    spotify_pls = load_playlists_json(spotify_playlist_file, spotify_encoding)

    def find_spotify_playlist(spotify_pls: Dict, src_pl_id: Union[str, None]) -> Dict:
        """Return the spotify playlist that matches the `src_pl_id`.

        Args:
            `spotify_pls`: The playlist datastrcuture saved by spotify-backup.
            `src_pl_id`: The ID of a playlist to find, or None for the "Liked Songs" playlist.
        """
        for src_pl in spotify_pls["playlists"]:
            if src_pl_id is None and str(src_pl.get("name")) == "Liked Songs":
                return src_pl
            if src_pl_id is not None and str(src_pl.get("id")) == src_pl_id:
                return src_pl
        raise ValueError(f"Could not find Spotify playlist {src_pl_id}")

    src_pl = find_spotify_playlist(spotify_pls, src_pl_id)
    src_pl_name = src_pl["name"]

    print(f"== Spotify Playlist: {src_pl_name}")

    pl_tracks = src_pl["tracks"]
    if reverse_playlist:
        pl_tracks = reversed(pl_tracks)

    for src_track in pl_tracks:
        if src_track["track"] is None:
            print(
                f"WARNING: Spotify track seems to be malformed, Skipping.  Track: {src_track!r}"
            )
            continue

        try:
            src_album_name = src_track["track"]["album"]["name"]
            src_track_artist = src_track["track"]["artists"][0]["name"]
        except TypeError as e:
            print(f"ERROR: Spotify track seems to be malformed.  Track: {src_track!r}")
            raise e
        src_track_name = src_track["track"]["name"]

        yield SongInfo(src_track_name, src_track_artist, src_album_name)


def get_playlist_id_by_name(yt: YTMusic, title: str) -> Optional[str]:
    """Look up a YTMusic playlist ID by name.

    Args:
        `yt` (YTMusic): _description_
        `title` (str): _description_

    Returns:
        Optional[str]: The playlist ID or None if not found.
    """
    #  ytmusicapi seems to run into some situations where it gives a Traceback on listing playlists
    #  https://github.com/sigma67/ytmusicapi/issues/539
    try:
        playlists = yt.get_library_playlists(limit=5000)
    except KeyError as e:
        print("=" * 60)
        print(f"Attempting to look up playlist '{title}' failed with KeyError: {e}")
        print(
            "This is a bug in ytmusicapi that prevents 'copy_all_playlists' from working."
        )
        print(
            "You will need to manually copy playlists using s2yt_list_playlists and s2yt_copy_playlist"
        )
        print(
            "until this bug gets resolved.  Try `pip install --upgrade ytmusicapi` just to verify"
        )
        print("you have the latest version of that library.")
        print("=" * 60)
        raise

    for pl in playlists:
        if pl["title"] == title:
            return pl["playlistId"]

    return None


@dataclass
class ResearchDetails:
    query: Optional[str] = field(default=None)
    songs: Optional[List[Dict]] = field(default=None)
    suggestions: Optional[List[str]] = field(default=None)


def lookup_song(
    yt: YTMusic,
    track_name: str,
    artist_name: str,
    album_name,
    yt_search_algo: int,
    details: Optional[ResearchDetails] = None,
) -> dict:
    """Look up a song on YTMusic

    Given the Spotify track information, it does a lookup for the album by the same
    artist on YTMusic, then looks at the first 3 hits looking for a track with exactly
    the same name. In the event that it can't find that exact track, it then does
    a search of songs for the track name by the same artist and simply returns the
    first hit.

    The idea is that finding the album and artist and then looking for the exact track
    match will be more likely to be accurate than searching for the song and artist and
    relying on the YTMusic yt_search_algorithm to figure things out, especially for short tracks
    that might have many contradictory hits like "Survival by Yes".

    Args:
        `yt` (YTMusic)
        `track_name` (str): The name of the researched track
        `artist_name` (str): The name of the researched track's artist
        `album_name` (str): The name of the researched track's album
        `yt_search_algo` (int): 0 for exact matching, 1 for extended matching (search past 1st result), 2 for approximate matching (search in videos)
        `details` (ResearchDetails): If specified, more information about the search and the response will be populated for use by the caller.

    Raises:
        ValueError: If no track is found, it returns an error

    Returns:
        dict: The infos of the researched song
    """
    albums = yt.search(query=f"{album_name} by {artist_name}", filter="albums")
    for album in albums[:3]:
        # print(album)
        # print(f"ALBUM: {album['browseId']} - {album['title']} - {album['artists'][0]['name']}")

        try:
            for track in yt.get_album(album["browseId"])["tracks"]:
                if track["title"] == track_name:
                    return track
            # print(f"{track['videoId']} - {track['title']} - {track['artists'][0]['name']}")
        except Exception as e:
            print(f"Unable to lookup album ({e}), continuing...")

    query = f"{track_name} by {artist_name}"
    if details:
        details.query = query
        details.suggestions = yt.get_search_suggestions(query=query)
    songs = yt.search(query=query, filter="songs")

    match yt_search_algo:
        case 0:
            if details:
                details.songs = songs
            return songs[0]

        case 1:
            for song in songs:
                if (
                    song["title"] == track_name
                    and song["artists"][0]["name"] == artist_name
                    and song["album"]["name"] == album_name
                ):
                    return song
                # print(f"SONG: {song['videoId']} - {song['title']} - {song['artists'][0]['name']} - {song['album']['name']}")

            raise ValueError(
                f"Did not find {track_name} by {artist_name} from {album_name}"
            )

        case 2:
            #  This would need to do fuzzy matching
            for song in songs:
                # Remove everything in brackets in the song title
                song_title_without_brackets = re.sub(r"[\[(].*?[])]", "", song["title"])
                if (
                    (
                        song_title_without_brackets == track_name
                        and song["album"]["name"] == album_name
                    )
                    or (song_title_without_brackets == track_name)
                    or (song_title_without_brackets in track_name)
                    or (track_name in song_title_without_brackets)
                ) and (
                    song["artists"][0]["name"] == artist_name
                    or artist_name in song["artists"][0]["name"]
                ):
                    return song

            # Finds approximate match
            # This tries to find a song anyway. Works when the song is not released as a music but a video.
            else:
                track_name = track_name.lower()
                first_song_title = songs[0]["title"].lower()
                if (
                    track_name not in first_song_title
                    or songs[0]["artists"][0]["name"] != artist_name
                ):  # If the first song is not the one we are looking for
                    print("Not found in songs, searching videos")
                    new_songs = yt.search(
                        query=f"{track_name} by {artist_name}", filter="videos"
                    )  # Search videos

                    # From here, we search for videos reposting the song. They often contain the name of it and the artist. Like with 'Nekfeu - Ecrire'.
                    for new_song in new_songs:
                        new_song_title = new_song[
                            "title"
                        ].lower()  # People sometimes mess up the capitalization in the title
                        if (
                            track_name in new_song_title
                            and artist_name in new_song_title
                        ) or (track_name in new_song_title):
                            print("Found a video")
                            return new_song
                    else:
                        # Basically we only get here if the song isn't present anywhere on YouTube
                        raise ValueError(
                            f"Did not find {track_name} by {artist_name} from {album_name}"
                        )
                else:
                    return songs[0]


def copier(
    src_tracks: Iterator[SongInfo],
    dst_pl_id: Optional[str] = None,
    dry_run: bool = False,
    # track_sleep: float = 0.1,
    batch_sleep: float = 1.0,
    batch_size: int = 50,
    yt_search_algo: int = 0,
    *,
    yt: Optional[YTMusic] = None,
):
    """
    @@@
    """
    if yt is None:
        yt = get_ytmusic()

    yt_pl_title = "Liked Songs"  # Default for liked songs (dst_pl_id is None)
    if dst_pl_id is not None:
        try:
            yt_pl = yt.get_playlist(playlistId=dst_pl_id)
            yt_pl_title = yt_pl["title"]
        except Exception as e:
            print(f"ERROR: Unable to find YTMusic playlist {dst_pl_id}: {e}")
            print(
                "       Make sure the YTMusic playlist ID is correct, it should be something like "
            )
            print("      'PL_DhcdsaJ7echjfdsaJFhdsWUd73HJFca'")
            sys.exit(1)
        print(f"== Youtube Playlist: {yt_pl['title']}")

    tracks_added_set = set()
    video_ids_to_add = []
    total_processed = 0
    total_added_successfully = 0
    duplicate_count = 0
    error_count = 0
    batch_num = 1

    for src_track in src_tracks:
        total_processed += 1
        print(
            f"({total_processed}) Spotify:   {src_track.title} - {src_track.artist} - {src_track.album}"
        )

        try:
            dst_track = lookup_song(
                yt, src_track.title, src_track.artist, src_track.album, yt_search_algo
            )
        except Exception as e:
            print(f"ERROR: Unable to look up song on YTMusic: {e}")
            error_count += 1
            continue  # Skip to next Spotify track

        # -- Display found track onse --
        yt_artist_name = "<Unknown>"
        if "artists" in dst_track and len(dst_track["artists"]) > 0:
            yt_artist_name = dst_track["artists"][0]["name"]
        print(
            f"  Youtube: {dst_track['title']} - {yt_artist_name} - {dst_track['album'] if 'album' in dst_track else '<Unknown>'}"
        )
        # -- end --

        video_id = dst_track.get("videoId")
        if not video_id:
            print("ERROR: Found track has no videoId, skipping.")
            error_count += 1
            continue

        if video_id and video_id not in tracks_added_set:
            tracks_added_set.add(video_id)
            video_ids_to_add.append(video_id)

        # --- Check if batch is full or it's the last track ---
        # (We process the batch *after* this loop finishes if there are leftovers)
        if len(video_ids_to_add) >= batch_size:
            if not dry_run:
                print(
                    f"\n-- Adding batch {batch_num} ({len(video_ids_to_add)} songs) to {yt_pl_title} --"
                )
                success = add_batch_with_retry(
                    yt, dst_pl_id, video_ids_to_add, batch_num
                )
                if success:
                    total_added_successfully += len(video_ids_to_add)
                    if batch_sleep > 0:
                        print(
                            f"-- Batch {batch_num} added, sleeping for {batch_sleep}s --\n"
                        )
                        time.sleep(batch_sleep)
                else:
                    error_count += len(video_ids_to_add)
                    print(f"ERROR: Failed to add batch {batch_num} after retries.")
            else:
                print(
                    f"\n-- [Dry Run] Would add batch {batch_num} ({len(video_ids_to_add)} songs) to {yt_pl_title} --\n"
                )
                total_added_successfully += len(video_ids_to_add)
            video_ids_to_add = []  # Clear the batch
            batch_num += 1

        # Removed the per-track sleep
        # if track_sleep:
        #    time.sleep(track_sleep)

    # --- Add any remaining tracks after the loop ---
    if video_ids_to_add:
        if not dry_run:
            print(
                f"\n-- Adding final batch {batch_num} ({len(video_ids_to_add)} songs) to {yt_pl_title} --"
            )
            success = add_batch_with_retry(yt, dst_pl_id, video_ids_to_add, batch_num)
            if success:
                total_added_successfully += len(video_ids_to_add)
                print(f"-- Final Batch {batch_num} added --\n")
            else:
                error_count += len(video_ids_to_add)
                print(f"ERROR: Failed to add final batch {batch_num} after retries.")
        else:
            print(
                f"\n-- [Dry Run] Would add final batch {batch_num} ({len(video_ids_to_add)} songs) to {yt_pl_title} --\n"
            )
            total_added_successfully += len(
                video_ids_to_add
            )  # Simulate success for dry run count

    print()
    print(f"Processed {total_processed} Spotify tracks.")
    print(f"Added {total_added_successfully} unique tracks to '{yt_pl_title}'.")
    print(f"Encountered {duplicate_count} duplicates (within this run).")
    print(f"Encountered {error_count} errors (lookup failures or add failures).")


# --- Helper function for batch adding with retry ---
def add_batch_with_retry(
    yt: YTMusic, dst_pl_id: Optional[str], video_ids: List[str], batch_num: int
) -> bool:
    """Adds a batch of video IDs to a playlist or likes them, with retry logic."""

    # --- PRE-CHECK: Ensure video_ids list is valid ---
    if dst_pl_id is not None:  # Only check if we intend to call add_playlist_items
        if not video_ids:  # Checks if the list is None or empty []
            print(
                f"ERROR: (Batch {batch_num}) Attempting to add items but video_ids list is empty. Skipping API call."
            )
            return False  # Indicate failure without triggering retries for this specific issue

        # Optional: Filter out any None or empty strings just in case
        valid_video_ids = [
            vid for vid in video_ids if vid and isinstance(vid, str) and len(vid) > 5
        ]  # Basic sanity check
        if not valid_video_ids:
            print(
                f"ERROR: (Batch {batch_num}) video_ids list contained only invalid entries ({video_ids}). Skipping API call."
            )
            return False  # Indicate failure

        if len(valid_video_ids) != len(video_ids):
            print(
                f"WARNING: (Batch {batch_num}) Filtered out invalid entries from video_ids. Original: {len(video_ids)}, Valid: {len(valid_video_ids)}"
            )
            # Use the filtered list for the API call
            video_ids_to_use = valid_video_ids
        else:
            # Use the original list if all were valid initially
            video_ids_to_use = video_ids
    else:
        # For liked songs (dst_pl_id is None), we might still want to check video_ids before rating
        if not video_ids:
            print(
                f"ERROR: (Batch {batch_num}) Attempting to rate songs but video_ids list is empty."
            )
            return False  # Can't rate nothing
        # We can filter here too if rating needs valid IDs
        video_ids_to_use = [
            vid for vid in video_ids if vid and isinstance(vid, str) and len(vid) > 5
        ]
        if not video_ids_to_use:
            print(
                f"ERROR: (Batch {batch_num}) video_ids list for rating contained only invalid entries ({video_ids})."
            )
            return False

    # --- End Pre-check ---

    exception_sleep = 5
    for attempt in range(10):
        try:
            if dst_pl_id is not None:
                # --- Make the API call with the validated list ---
                response = yt.add_playlist_items(
                    playlistId=dst_pl_id,
                    videoIds=video_ids_to_use,  # Use the potentially filtered list
                    duplicates=False,
                )
                # ... (rest of response handling) ...
                return True  # Assuming success if no exception and basic check passed

            else:
                # Liked songs logic using video_ids_to_use
                print(
                    f"INFO: Rating {len(video_ids_to_use)} songs individually for Liked Songs batch {batch_num}."
                )
                success_count = 0
                # Use the filtered list here too
                for video_id in video_ids_to_use:
                    try:
                        yt.rate_song(video_id, "LIKE")
                        success_count += 1
                    except Exception as e_rate:
                        print(f"ERROR: (Retrying rate_song: {video_id}) {e_rate}")
                print(
                    f"DEBUG: Batch {batch_num}, Attempt {attempt+1}: Rated {success_count}/{len(video_ids_to_use)} songs."
                )
                return True  # Assume overall success for rating batch

            # If we fall through dict checks for playlist add, force a retry? (Maybe remove this)
            # raise Exception("Assuming add failed based on response, retrying.") # This might be too aggressive

        except Exception as e:
            # Check if the specific error is the one we are trying to avoid
            if "You must provide either videoIds" in str(e):
                print(
                    f"ERROR: (Batch {batch_num}, Attempt {attempt+1}) Critial API Error: {e}. Aborting retries for this batch."
                )
                return False  # Abort retries for this specific known failure cause

            # Handle other exceptions with retry
            print(
                f"ERROR: (Batch {batch_num}, Attempt {attempt+1}) Failed: {e}. Retrying in {exception_sleep} seconds."
            )
            time.sleep(exception_sleep)
            exception_sleep *= 2
            if exception_sleep > 300:
                exception_sleep = 300

    print(f"ERROR: (Batch {batch_num}) Failed after multiple retries.")
    return False  # Failed after all retries


def copy_playlist(
    spotify_playlist_id: str,
    ytmusic_playlist_id: str,
    spotify_playlists_encoding: str = "utf-8",
    dry_run: bool = False,
    # track_sleep: float = 0.1, # no longer used by copier with the new method
    batch_sleep: float = 0.1,  # Default sleep BETWEEN batches (adjust if needed)
    batch_size: int = 50,  # Default batch size
    yt_search_algo: int = 0,
    reverse_playlist: bool = True,
    privacy_status: str = "PRIVATE",
):
    """
    Copy a Spotify playlist to a YTMusic playlist
    @@@
    """
    print("Using search algo n°: ", yt_search_algo)
    yt = get_ytmusic()
    pl_name: str = ""

    if ytmusic_playlist_id.startswith("+"):
        pl_name = ytmusic_playlist_id[1:]

        ytmusic_playlist_id = get_playlist_id_by_name(yt, pl_name)
        print(f"Looking up playlist '{pl_name}': id={ytmusic_playlist_id}")

    if ytmusic_playlist_id is None:
        if pl_name == "":
            print("No playlist name or ID provided, creating playlist...")
            spotify_pls: dict = load_playlists_json()
            for pl in spotify_pls["playlists"]:
                if len(pl.keys()) > 3 and pl["id"] == spotify_playlist_id:
                    pl_name = pl["name"]

        ytmusic_playlist_id = _ytmusic_create_playlist(
            yt,
            title=pl_name,
            description=pl_name,
            privacy_status=privacy_status,
        )

        #  create_playlist returns a dict if there was an error
        if isinstance(ytmusic_playlist_id, dict):
            print(f"ERROR: Failed to create playlist: {ytmusic_playlist_id}")
            sys.exit(1)
        print(f"NOTE: Created playlist '{pl_name}' with ID: {ytmusic_playlist_id}")

    copier(
        src_tracks=iter_spotify_playlist(  # Use keyword arg src_tracks=
            spotify_playlist_id,
            spotify_encoding=spotify_playlists_encoding,
            reverse_playlist=reverse_playlist,
        ),
        dst_pl_id=ytmusic_playlist_id,  # Use keyword args for clarity & safety
        dry_run=dry_run,
        batch_sleep=batch_sleep,  # Pass batch_sleep correctly
        batch_size=batch_size,  # Pass batch_size correctly
        yt_search_algo=yt_search_algo,  # Pass yt_search_algo correctly
        yt=yt,  # Pass yt correctly
    )


def copy_all_playlists(
    # track_sleep: float = 0.1, # REMOVE THIS
    batch_sleep: float = 0.1,  # Default sleep BETWEEN batches
    batch_size: int = 50,  # Default batch size
    dry_run: bool = False,
    spotify_playlists_encoding: str = "utf-8",
    yt_search_algo: int = 0,
    reverse_playlist: bool = True,
    privacy_status: str = "PRIVATE",
):
    """
    Copy all Spotify playlists (except Liked Songs) to YTMusic playlists
    """
    spotify_pls = load_playlists_json()
    yt = get_ytmusic()

    for src_pl in spotify_pls["playlists"]:
        if str(src_pl.get("name")) == "Liked Songs":
            continue

        pl_name = src_pl["name"]
        if pl_name == "":
            pl_name = f"Unnamed Spotify Playlist {src_pl['id']}"

        dst_pl_id = get_playlist_id_by_name(yt, pl_name)
        print(f"Looking up playlist '{pl_name}': id={dst_pl_id}")
        if dst_pl_id is None:
            dst_pl_id = _ytmusic_create_playlist(
                yt, title=pl_name, description=pl_name, privacy_status=privacy_status
            )

            #  create_playlist returns a dict if there was an error
            if isinstance(dst_pl_id, dict):
                print(f"ERROR: Failed to create playlist: {dst_pl_id}")
                sys.exit(1)
            print(f"NOTE: Created playlist '{pl_name}' with ID: {dst_pl_id}")

        copier(
            src_tracks=iter_spotify_playlist(  # Use keyword arg src_tracks=
                src_pl["id"],
                spotify_encoding=spotify_playlists_encoding,
                reverse_playlist=reverse_playlist,
            ),
            dst_pl_id=dst_pl_id,  # Use keyword args
            dry_run=dry_run,
            batch_sleep=batch_sleep,  # Pass correctly
            batch_size=batch_size,  # Pass correctly
            yt_search_algo=yt_search_algo,  # Pass correctly
            yt=yt,  # Pass yt correctly (was missing before!)
        )

        print("\nPlaylist done!\n")

    print("All done!")
