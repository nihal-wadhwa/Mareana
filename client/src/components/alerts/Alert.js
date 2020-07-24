import React from 'react';

const Alert = ({ docs, symbols }) => (
    <div>
        <h3> <i className=" material-icons">done</i>{`Total of ${docs} documents were annotated`}</h3 >
        <h3><i className="material-icons">done</i>{`Total of ${symbols} symbols were classified`}</h3>
    </div >
)

export default Alert;