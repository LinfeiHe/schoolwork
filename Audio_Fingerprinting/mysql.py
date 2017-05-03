import MySQLdb
import sqlite3
from itertools import izip_longest

class SQLDatabase(object):
    # tables
    FINGERPRINTS_TABLE_NAME = "fingerprints"
    SONGS_TABLE_NAME = "songs"

    # fields
    FIELD_HASH = "hash"
    FIELD_SONG_ID = "song_id"
    FIELD_OFFSET = "offset"
    FIELD_SONG_NAME = "song_name"
    FIELD_FINGERPRINTED = "fingerprinted"

    # creates
    CREATE_FINGERPRINTS_TABLE = """
            CREATE TABLE IF NOT EXISTS `%s` (
                 `%s` binary(10) not null,
                 `%s` mediumint unsigned not null,
                 `%s` int unsigned not null,
             INDEX (%s),
             UNIQUE KEY `unique_constraint` (%s, %s, %s),
             FOREIGN KEY (%s) REFERENCES %s(%s) ON DELETE CASCADE
        ) ENGINE=INNODB;""" % (
        FINGERPRINTS_TABLE_NAME, FIELD_HASH,
        FIELD_SONG_ID, FIELD_OFFSET, FIELD_HASH,
        FIELD_SONG_ID, FIELD_OFFSET, FIELD_HASH,
        FIELD_SONG_ID, SONGS_TABLE_NAME, FIELD_SONG_ID
    )

    CREATE_SONGS_TABLE = """
            CREATE TABLE IF NOT EXISTS `%s` (
                `%s` mediumint unsigned not null auto_increment,
                `%s` varchar(250) not null,
                `%s` tinyint default 0,
            PRIMARY KEY (`%s`),
            UNIQUE KEY `%s` (`%s`)
        ) ENGINE=INNODB;""" % (
        SONGS_TABLE_NAME, FIELD_SONG_ID, FIELD_SONG_NAME, FIELD_FINGERPRINTED,
        FIELD_SONG_ID, FIELD_SONG_ID, FIELD_SONG_ID,
    )

    CREATE_FINGERPRINTS_TABLE_LITE = """
                CREATE TABLE IF NOT EXISTS `%s` (
                     `%s` binary(10) not null UNIQUE,
                     `%s` mediumint unsigned not null UNIQUE,
                     `%s` int unsigned not null UNIQUE,
                 UNIQUE (%s, %s, %s) ON CONFLICT REPLACE,
                 FOREIGN KEY (%s) REFERENCES %s(%s) ON DELETE CASCADE
            );""" % (
        FINGERPRINTS_TABLE_NAME,
        FIELD_SONG_ID, FIELD_OFFSET, FIELD_HASH,
        FIELD_SONG_ID, FIELD_OFFSET, FIELD_HASH,
        FIELD_SONG_ID, SONGS_TABLE_NAME, FIELD_SONG_ID
    )

    CREATE_SONGS_TABLE_LITE = """
                CREATE TABLE IF NOT EXISTS `%s` (
                    `%s` INTEGER PRIMARY KEY AUTOINCREMENT UNIQUE,
                    `%s` varchar(250) not null,
                    `%s` tinyint default 0
            );""" % (
        SONGS_TABLE_NAME, FIELD_SONG_ID, FIELD_SONG_NAME, FIELD_FINGERPRINTED
    )

    # inserts (ignores duplicates)
    INSERT_FINGERPRINT = """
            INSERT IGNORE INTO %s (%s, %s, %s) values
                (UNHEX(%%s), %%s, %%s);
        """ % (FINGERPRINTS_TABLE_NAME, FIELD_HASH, FIELD_SONG_ID, FIELD_OFFSET)

    INSERT_SONG = "INSERT INTO %s (%s) values (%%s);" % (
        SONGS_TABLE_NAME, FIELD_SONG_NAME)

    # selects
    SELECT = """
            SELECT %s, %s FROM %s WHERE %s = UNHEX(%%s);
        """ % (FIELD_SONG_ID, FIELD_OFFSET, FINGERPRINTS_TABLE_NAME, FIELD_HASH)

    SELECT_MULTIPLE = """
            SELECT HEX(%s), %s, %s FROM %s WHERE %s IN (%%s);
        """ % (FIELD_HASH, FIELD_SONG_ID, FIELD_OFFSET,
               FINGERPRINTS_TABLE_NAME, FIELD_HASH)

    SELECT_ALL = """
            SELECT %s, %s FROM %s;
        """ % (FIELD_SONG_ID, FIELD_OFFSET, FINGERPRINTS_TABLE_NAME)

    SELECT_SONG = """
            SELECT %s FROM %s WHERE %s = %%s
        """ % (FIELD_SONG_NAME, SONGS_TABLE_NAME, FIELD_SONG_ID)

    SELECT_NUM_FINGERPRINTS = """
            SELECT COUNT(*) as n FROM %s
        """ % (FINGERPRINTS_TABLE_NAME)

    SELECT_UNIQUE_SONG_IDS = """
            SELECT COUNT(DISTINCT %s) as n FROM %s WHERE %s = 1;
        """ % (FIELD_SONG_ID, SONGS_TABLE_NAME, FIELD_FINGERPRINTED)

    SELECT_SONGS = """
            SELECT %s, %s FROM %s WHERE %s = 1;
        """ % (FIELD_SONG_ID, FIELD_SONG_NAME, SONGS_TABLE_NAME, FIELD_FINGERPRINTED)

    # drops
    DROP_FINGERPRINTS = "DROP TABLE IF EXISTS %s;" % FINGERPRINTS_TABLE_NAME
    DROP_SONGS = "DROP TABLE IF EXISTS %s;" % SONGS_TABLE_NAME

    # update
    UPDATE_SONG_FINGERPRINTED = """
            UPDATE %s SET %s = 1 WHERE %s = %%s
        """ % (SONGS_TABLE_NAME, FIELD_FINGERPRINTED, FIELD_SONG_ID)

    # delete
    DELETE_UNFINGERPRINTED = """
            DELETE FROM %s WHERE %s = 0;
        """ % (SONGS_TABLE_NAME, FIELD_FINGERPRINTED)

    def __init__(self, db_type='mysql'):
        self.db_type = db_type
        if db_type == 'mysql':
            self.conn = MySQLdb.connect(
                host='172.21.220.150',
                port=3306,
                user='root',
                passwd='123',
                db='test'
            )
        elif db_type == 'sqlite3':
            self.conn = sqlite3.connect('project.db')
        self.cur = self.conn.cursor()

    def __del__(self):
        self.cur.close()
        self.conn.close()

    def create_table(self):
        if self.db_type == 'mysql':
            self.cur.execute(self.CREATE_SONGS_TABLE)
            self.cur.execute(self.CREATE_FINGERPRINTS_TABLE)
            self.conn.commit()
        elif self.db_type == 'sqlite3':
            self.cur.execute(self.CREATE_SONGS_TABLE_LITE)
            self.cur.execute(self.CREATE_FINGERPRINTS_TABLE_LITE)
            self.conn.commit()

    def insert_song(self, song_name):
        """
        Inserts song in the database and returns the ID of the inserted record.
        """
        self.cur.execute(self.INSERT_SONG, (song_name,))
        sid = self.cur.lastrowid
        self.cur.execute(self.UPDATE_SONG_FINGERPRINTED, (sid,))
        self.conn.commit()
        return sid

    def get_song_by_id(self, sid):
        """
        Returns song by its ID.
        """
        self.cur = self.conn.cursor(MySQLdb.cursors.DictCursor)
        self.cur.execute(self.SELECT_SONG, (sid,))
        return self.cur.fetchone()

    def insert_fingerprints(self, sid, hash_):
        """
        Insert series of hash => song_id, offset
        values into the database.
        """
        values = []
        for hashes, offset in hash_:
            values.append((hashes, sid, offset))

        for split_values in grouper(values, 1000):
            self.cur.executemany(self.INSERT_FINGERPRINT, split_values)
            self.conn.commit()

    def return_matches(self, hashes):
        """
        Return the (song_id, offset_diff) tuples associated with
        a list of (sha1, sample_offset) values.
        """
        # Create a dictionary of hash => offset pairs for later lookups
        mapper = {}
        for hash_, offset in hashes:
            mapper[hash_.upper()] = offset

        # Get an iteratable of all the hashes we need
        values = mapper.keys()

        for split_values in grouper(values, 1000):
            # Create our IN part of the query
            query = self.SELECT_MULTIPLE
            query = query % ', '.join(['UNHEX(%s)'] * len(split_values))

            self.cur.execute(query, split_values)

            for hash, sid, offset in self.cur:
                # (sid, db_offset - song_sampled_offset)
                yield (sid, offset - mapper[hash])

    def align_matches(self, matches):
        """
            Finds hash matches that align in time with other matches and finds
            consensus about which hashes are "true" signal from the audio.

            Returns a dictionary with match information.
        """
        # align by diffs
        diff_counter = {}
        largest = 0
        largest_count = 0
        song_id = -1
        for tup in matches:
            sid, diff = tup
            if not diff in diff_counter:
                diff_counter[diff] = {}
            if not sid in diff_counter[diff]:
                diff_counter[diff][sid] = 0
            diff_counter[diff][sid] += 1

            if diff_counter[diff][sid] > largest_count:
                largest = diff
                largest_count = diff_counter[diff][sid]
                song_id = sid

        print("Diff is %d with %d offset-aligned matches" % (largest,
                                                             largest_count))

        # extract idenfication
        song = self.get_song_by_id(song_id)
        if song:
            songname = song.get("song_name", None)
        else:
            return None

        # return match info
        song = {
            "song_id": song_id,
            "song_name": songname,
            "confidence": largest_count,
            "offset": largest
        }

        return song

def grouper(iterable, n, fillvalue=None):
    args = [iter(iterable)] * n
    return (filter(None, values) for values
            in izip_longest(fillvalue=fillvalue, *args))

if __name__ == "__main__":
    db = SQLDatabase()
    db.create_table()
